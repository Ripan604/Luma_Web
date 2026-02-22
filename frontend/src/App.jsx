import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { FaceMesh } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";
import { Hands } from "@mediapipe/hands";

// Eye Aspect Ratio (EAR): lower = more closed. Returns { left, right, avg }.
function getEAR(landmarks) {
  const d = (a, b) =>
    Math.hypot(landmarks[a].x - landmarks[b].x, landmarks[a].y - landmarks[b].y);
  const leftVert1 = d(159, 145);
  const leftVert2 = d(158, 153);
  const leftHoriz = d(33, 133) || 0.001;
  const earLeft = (leftVert1 + leftVert2) / (2 * leftHoriz);
  const rightVert1 = d(386, 374);
  const rightVert2 = d(385, 380);
  const rightHoriz = d(362, 263) || 0.001;
  const earRight = (rightVert1 + rightVert2) / (2 * rightHoriz);
  return { left: earLeft, right: earRight, avg: (earLeft + earRight) / 2 };
}

const EAR_CLOSED = 0.22;   // Eyes considered closed
const EAR_OPEN = 0.255;    // Eyes considered open after a blink
const BLINK_COOLDOWN_MS = 220;  // Min ms between blinks
const EAR_SMOOTH_SAMPLES = 4;   // Running average for stability
const FINGER_SMOOTH_SAMPLES = 7; // Majority vote window for finger count
const FINGER_STABLE_MS = 550;   // Must hold correct count this long
const FACE_DETECTION_MIN_CONFIDENCE = 0.42;
const FACE_TRACKING_MIN_CONFIDENCE = 0.36;
const FACE_MISSING_FRAME_TOLERANCE = 8;
const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 480;
const CAMERA_FRAME_RATE_IDEAL = 24;
const CAMERA_BRIGHTNESS_MIN = 70;
const CAMERA_BRIGHTNESS_MAX = 180;
const CAMERA_CONTRAST_MIN = 70;
const CAMERA_CONTRAST_MAX = 180;
const CAMERA_FILTER_DEFAULT = { brightness: 100, contrast: 100 };
const IDENTITY_SIMILARITY_MIN = 0.9;
const IDENTITY_CENTER_SHIFT_MAX = 0.24;
const IDENTITY_AREA_RATIO_MIN = 0.45;
const IDENTITY_AREA_RATIO_MAX = 2.2;
const IDENTITY_MISMATCH_MAX = 12;
const IDENTITY_MISSING_MAX = 15;
const FACE_SIGNATURE_POINTS = [33, 263, 1, 61, 291, 4, 10, 152, 70, 300, 234, 454];
const TYPING_IDENTITY_SIMILARITY_MIN = 0.92;
const TYPING_CENTER_SHIFT_MAX = 0.24;
const TYPING_AREA_RATIO_MIN = 0.5;
const TYPING_AREA_RATIO_MAX = 1.95;

function dist3(a, b) {
  const dzWeight = 0.65;
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z - b.z) * dzWeight);
}

function angleBetween(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const mag1 = Math.hypot(v1.x, v1.y, v1.z);
  const mag2 = Math.hypot(v2.x, v2.y, v2.z);
  if (mag1 < 1e-6 || mag2 < 1e-6) return 0;
  const cos = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function angleAt(a, b, c) {
  const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  return angleBetween(ba, bc);
}

function averagePoint(points) {
  if (!points || points.length === 0) return { x: 0, y: 0, z: 0 };
  const sum = points.reduce(
    (acc, p) => ({
      x: acc.x + p.x,
      y: acc.y + p.y,
      z: acc.z + p.z,
    }),
    { x: 0, y: 0, z: 0 }
  );
  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
    z: sum.z / points.length,
  };
}

// Count extended fingers using joint straightness + reach + thumb spread checks.
function countExtendedFingers(landmarks, handednessLabel = "") {
  if (!landmarks || landmarks.length < 21) return 0;

  const wrist = landmarks[0];
  const handScale = Math.max(
    dist3(landmarks[5], landmarks[17]),
    dist3(landmarks[0], landmarks[9])
  );
  if (handScale < 0.06) return 0;

  const distFromWrist = (idx) => dist3(landmarks[idx], wrist);
  const isFingerExtended = (mcp, pip, dip, tip, minReachRatio = 1.05) => {
    const pipAngle = angleAt(landmarks[mcp], landmarks[pip], landmarks[dip]);
    const dipAngle = angleAt(landmarks[pip], landmarks[dip], landmarks[tip]);
    const reachRatio = distFromWrist(tip) / Math.max(distFromWrist(pip), 0.0001);
    return pipAngle > 160 && dipAngle > 150 && reachRatio > minReachRatio;
  };

  let count = 0;
  if (isFingerExtended(5, 6, 7, 8, 1.05)) count += 1; // Index
  if (isFingerExtended(9, 10, 11, 12, 1.06)) count += 1; // Middle
  if (isFingerExtended(13, 14, 15, 16, 1.04)) count += 1; // Ring
  if (isFingerExtended(17, 18, 19, 20, 1.03)) count += 1; // Pinky

  const thumbIpAngle = angleAt(landmarks[2], landmarks[3], landmarks[4]);
  const thumbMcpAngle = angleAt(landmarks[1], landmarks[2], landmarks[3]);
  const thumbReachRatio = distFromWrist(4) / Math.max(distFromWrist(2), 0.0001);
  const palmCenter = averagePoint([
    landmarks[0],
    landmarks[5],
    landmarks[9],
    landmarks[13],
    landmarks[17],
  ]);
  const thumbPalmRatio =
    dist3(landmarks[4], palmCenter) / Math.max(dist3(landmarks[3], palmCenter), 0.0001);
  const thumbSpreadAngle = angleBetween(
    {
      x: landmarks[4].x - landmarks[0].x,
      y: landmarks[4].y - landmarks[0].y,
      z: landmarks[4].z - landmarks[0].z,
    },
    {
      x: landmarks[5].x - landmarks[0].x,
      y: landmarks[5].y - landmarks[0].y,
      z: landmarks[5].z - landmarks[0].z,
    }
  );

  let thumbExtended =
    thumbIpAngle > 150 &&
    thumbMcpAngle > 145 &&
    thumbReachRatio > 1.15 &&
    thumbPalmRatio > 1.08 &&
    thumbSpreadAngle > 15;

  // Handedness hint improves edge cases where thumb is half-open.
  if (!thumbExtended && thumbReachRatio > 1.1) {
    const thumbDir = landmarks[4].x - landmarks[3].x;
    if (handednessLabel === "Right" && thumbDir < -handScale * 0.08) {
      thumbExtended = true;
    } else if (handednessLabel === "Left" && thumbDir > handScale * 0.08) {
      thumbExtended = true;
    }
  }
  if (thumbExtended) count += 1;

  return Math.min(5, Math.max(0, count));
}

function getMajorityFingerCount(history) {
  if (!history || history.length === 0) return null;
  const recent = history.slice(-FINGER_SMOOTH_SAMPLES);
  const counts = {};
  for (const c of recent) {
    counts[c] = (counts[c] || 0) + 1;
  }
  let best = null;
  let bestCount = -1;
  for (const [count, num] of Object.entries(counts)) {
    if (num > bestCount) {
      bestCount = num;
      best = parseInt(count, 10);
    }
  }
  return best;
}

// Majority vote from recent counts for stability
function smoothFingerCount(history, required) {
  if (history.length < FINGER_SMOOTH_SAMPLES) return null;
  const majority = getMajorityFingerCount(history);
  return majority === required ? majority : null;
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

function buildFaceIdentitySnapshot(landmarks) {
  if (!landmarks || landmarks.length < 468) return null;
  const dist = (a, b) =>
    Math.hypot(landmarks[a].x - landmarks[b].x, landmarks[a].y - landmarks[b].y);
  const eyeDist = dist(33, 263) || 0.0001;
  const nose = landmarks[1];

  const descriptor = [];
  for (let i = 0; i < FACE_SIGNATURE_POINTS.length; i++) {
    const idx = FACE_SIGNATURE_POINTS[i];
    descriptor.push((landmarks[idx].x - nose.x) / eyeDist);
    descriptor.push((landmarks[idx].y - nose.y) / eyeDist);
  }
  descriptor.push(dist(61, 291) / eyeDist);
  descriptor.push(dist(10, 152) / eyeDist);
  descriptor.push(dist(70, 300) / eyeDist);

  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (let i = 0; i < landmarks.length; i++) {
    const p = landmarks[i];
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }

  return {
    descriptor,
    center: {
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
    },
    area: Math.max((maxX - minX) * (maxY - minY), 0.00001),
  };
}

function cloneIdentitySnapshot(snapshot) {
  if (!snapshot) return null;
  return {
    descriptor: [...snapshot.descriptor],
    center: {
      x: snapshot.center.x,
      y: snapshot.center.y,
    },
    area: snapshot.area,
  };
}

function isIdentityMatch(
  reference,
  current,
  {
    similarityMin = IDENTITY_SIMILARITY_MIN,
    centerShiftMax = IDENTITY_CENTER_SHIFT_MAX,
    areaRatioMin = IDENTITY_AREA_RATIO_MIN,
    areaRatioMax = IDENTITY_AREA_RATIO_MAX,
  } = {}
) {
  if (!reference || !current) return false;
  const similarity = cosineSimilarity(reference.descriptor, current.descriptor);
  const centerShift = Math.hypot(
    reference.center.x - current.center.x,
    reference.center.y - current.center.y
  );
  const areaRatio = current.area / (reference.area || current.area);
  return (
    similarity >= similarityMin &&
    centerShift <= centerShiftMax &&
    areaRatio >= areaRatioMin &&
    areaRatio <= areaRatioMax
  );
}

function App() {
  const [content, setContent] = useState("");
  const [keystrokes, setKeystrokes] = useState([]);
  const [cameraOn, setCameraOn] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState({ verify: false, collect: false, demo: false });
  const [toast, setToast] = useState({ show: false, message: "", type: "success" });
  const [faceDetected, setFaceDetected] = useState(false);
  const [multiFaceDetected, setMultiFaceDetected] = useState(false);
  const [cameraFilter, setCameraFilter] = useState(CAMERA_FILTER_DEFAULT);
  const [cameraTuningOpen, setCameraTuningOpen] = useState(false);
  const faceVectorRef = useRef([]);
  const latestFaceIdentityRef = useRef(null);
  const typingIdentityRef = useRef(null);
  const faceDetectedRef = useRef(false);
  const multiFaceDetectedRef = useRef(false);
  const noFaceFramesRef = useRef(0);
  const cameraFilterRef = useRef(CAMERA_FILTER_DEFAULT);
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "dark";
    const saved = window.localStorage.getItem("luma-theme");
    if (saved === "dark" || saved === "light") return saved;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  });
  const sceneRef = useRef(null);
  const particleCanvasRef = useRef(null);
  const particleCtxRef = useRef(null);
  const particlesRef = useRef([]);
  const frameClockRef = useRef(0);
  const reducedMotionRef = useRef(false);
  const pointerTargetRef = useRef({ x: 0, y: 0, active: false });
  const pointerCurrentRef = useRef({ x: 0, y: 0, active: false });
  const pointerAnimRef = useRef(null);
  const pointerMetaRef = useRef({ width: 1, height: 1 });

  // Liveness: require blink (and optional movement) before verify/collect
  const [livenessActive, setLivenessActive] = useState(false);
  const [livenessCount, setLivenessCount] = useState(0);
  const [livenessRequired, setLivenessRequired] = useState(2);
  const [livenessPrompt, setLivenessPrompt] = useState("");
  const [livenessChallengeType, setLivenessChallengeType] = useState("blink"); // 'blink' | 'fingers'
  const livenessRef = useRef({
    active: false,
    required: 2,
    count: 0,
    eyesClosed: false,
    lastBlinkTime: 0,
    onComplete: null,
    challengeType: null,
    fingerCountHistory: [],
    fingerStableSince: null,
    earHistory: [],
    identityRef: null,
    mismatchFrames: 0,
    missingFrames: 0,
  });
  const setLivenessCountRef = useRef(() => {});

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const faceMeshRef = useRef(null);
  const handsRef = useRef(null);
  const processingCanvasRef = useRef(null);
  const processingCtxRef = useRef(null);
  const [fingersShown, setFingersShown] = useState(0);
  const setFingersShownRef = useRef(() => {});

  const setFaceDetectedSafe = (next) => {
    if (faceDetectedRef.current !== next) {
      faceDetectedRef.current = next;
      setFaceDetected(next);
    }
  };

  const setMultiFaceDetectedSafe = (next) => {
    if (multiFaceDetectedRef.current !== next) {
      multiFaceDetectedRef.current = next;
      setMultiFaceDetected(next);
    }
  };

  useEffect(() => {
    cameraFilterRef.current = cameraFilter;
  }, [cameraFilter]);

  const updateCameraFilter = (key, value) => {
    setCameraFilter((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const resetCameraFilter = () => {
    setCameraFilter(CAMERA_FILTER_DEFAULT);
  };

  const closeCameraTuning = () => {
    setCameraTuningOpen(false);
  };

  const toggleCameraTuning = () => {
    setCameraTuningOpen((prev) => !prev);
  };

  const failLivenessIdentity = (message) => {
    livenessRef.current.active = false;
    livenessRef.current.challengeType = null;
    livenessRef.current.onComplete = null;
    livenessRef.current.identityRef = null;
    livenessRef.current.mismatchFrames = 0;
    livenessRef.current.missingFrames = 0;
    livenessRef.current.fingerStableSince = null;
    livenessRef.current.fingerCountHistory = [];
    livenessRef.current.earHistory = [];
    setLivenessActive(false);
    setLivenessCount(0);
    setFingersShown(0);
    showToast(message, "error");
  };

  const validateTypingIdentity = () => {
    if (!typingIdentityRef.current) {
      showToast(
        "Type while exactly one face is visible, then verify with the same person.",
        "error"
      );
      return false;
    }
    const current = latestFaceIdentityRef.current;
    if (!current || !faceDetectedRef.current || multiFaceDetectedRef.current) {
      showToast("Keep exactly one face visible before verifying.", "error");
      return false;
    }
    const match = isIdentityMatch(typingIdentityRef.current, current, {
      similarityMin: TYPING_IDENTITY_SIMILARITY_MIN,
      centerShiftMax: TYPING_CENTER_SHIFT_MAX,
      areaRatioMin: TYPING_AREA_RATIO_MIN,
      areaRatioMax: TYPING_AREA_RATIO_MAX,
    });
    if (!match) {
      typingIdentityRef.current = null;
      setKeystrokes([]);
      showToast(
        "Face changed after typing. Keystrokes were reset. Type again with the same person.",
        "error"
      );
      return false;
    }
    return true;
  };

  const requestCameraStream = async () => {
    const preferredConstraints = {
      facingMode: { ideal: "user" },
      width: { ideal: CAMERA_WIDTH },
      height: { ideal: CAMERA_HEIGHT },
      frameRate: { ideal: CAMERA_FRAME_RATE_IDEAL, max: 30 },
    };

    try {
      return await navigator.mediaDevices.getUserMedia({
        video: preferredConstraints,
        audio: false,
      });
    } catch {
      return await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
  };

  const optimizeVideoTrackForLowLight = async (stream) => {
    const track = stream?.getVideoTracks?.()[0];
    if (!track || !track.getCapabilities || !track.applyConstraints) return;

    const caps = track.getCapabilities();
    const advanced = {};

    if (Array.isArray(caps.exposureMode) && caps.exposureMode.includes("continuous")) {
      advanced.exposureMode = "continuous";
    }
    if (Array.isArray(caps.whiteBalanceMode) && caps.whiteBalanceMode.includes("continuous")) {
      advanced.whiteBalanceMode = "continuous";
    }
    if (Array.isArray(caps.focusMode) && caps.focusMode.includes("continuous")) {
      advanced.focusMode = "continuous";
    }
    if (caps.exposureCompensation?.max !== undefined && caps.exposureCompensation?.min !== undefined) {
      const min = caps.exposureCompensation.min;
      const max = caps.exposureCompensation.max;
      const boosted = min + (max - min) * 0.7;
      advanced.exposureCompensation = Math.max(min, Math.min(max, boosted));
    }
    if (caps.brightness?.max !== undefined && caps.brightness?.min !== undefined) {
      const min = caps.brightness.min;
      const max = caps.brightness.max;
      const boosted = min + (max - min) * 0.65;
      advanced.brightness = Math.max(min, Math.min(max, boosted));
    }

    if (Object.keys(advanced).length === 0) return;

    try {
      await track.applyConstraints({ advanced: [advanced] });
    } catch (error) {
      console.warn("Low-light track optimization not fully supported:", error);
    }
  };

  const getCameraInputFrame = () => {
    const video = videoRef.current;
    if (!video) return null;

    const { brightness, contrast } = cameraFilterRef.current;
    const needsFilter =
      brightness !== CAMERA_FILTER_DEFAULT.brightness ||
      contrast !== CAMERA_FILTER_DEFAULT.contrast;

    if (!needsFilter) {
      return video;
    }

    let canvas = processingCanvasRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      processingCanvasRef.current = canvas;
    }

    const width = video.videoWidth || CAMERA_WIDTH;
    const height = video.videoHeight || CAMERA_HEIGHT;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    let ctx = processingCtxRef.current;
    if (!ctx) {
      ctx = canvas.getContext("2d");
      processingCtxRef.current = ctx;
    }
    if (!ctx) return video;

    ctx.clearRect(0, 0, width, height);
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
    ctx.drawImage(video, 0, 0, width, height);
    ctx.filter = "none";

    return canvas;
  };

  // ---------- Start Camera ----------
  const startCamera = async () => {
    try {
      const stream = await requestCameraStream();
      await optimizeVideoTrackForLowLight(stream);
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      noFaceFramesRef.current = 0;
      setCameraOn(true);

      const faceMesh = new FaceMesh({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 2,
        refineLandmarks: true,
        minDetectionConfidence: FACE_DETECTION_MIN_CONFIDENCE,
        minTrackingConfidence: FACE_TRACKING_MIN_CONFIDENCE,
      });

      const hands = new Hands({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });
      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.65,
        minTrackingConfidence: 0.55,
      });
      hands.onResults((results) => {
        const l = livenessRef.current;
        if (!l.active || l.challengeType !== "fingers") return;
        if (results.multiHandLandmarks?.length > 0) {
          const handScore = results.multiHandedness?.[0]?.score ?? 1;
          if (handScore < 0.55) {
            l.fingerStableSince = null;
            setFingersShownRef.current?.(0);
            return;
          }
          const handednessLabel = results.multiHandedness?.[0]?.label || "";
          const count = countExtendedFingers(results.multiHandLandmarks[0], handednessLabel);
          l.fingerCountHistory = (l.fingerCountHistory || []).slice(-FINGER_SMOOTH_SAMPLES);
          l.fingerCountHistory.push(count);
          const majority = getMajorityFingerCount(l.fingerCountHistory);
          setFingersShownRef.current?.(majority ?? count);

          const smoothed = smoothFingerCount(l.fingerCountHistory, l.required);
          const now = Date.now();
          if (smoothed !== null) {
            if (!l.fingerStableSince) {
              l.fingerStableSince = now;
            } else if (now - l.fingerStableSince >= FINGER_STABLE_MS) {
              l.active = false;
              l.challengeType = null;
              l.identityRef = null;
              l.mismatchFrames = 0;
              l.missingFrames = 0;
              l.fingerStableSince = null;
              l.fingerCountHistory = [];
              const done = l.onComplete;
              l.onComplete = null;
              if (done) done();
            }
          } else {
            // Count doesn't match - reset stability timer
            l.fingerStableSince = null;
          }
        } else {
          // No hand detected - reset
          l.fingerStableSince = null;
        }
      });
      handsRef.current = hands;

      faceMesh.onResults((results) => {
        const faces = results.multiFaceLandmarks || [];
        const faceCount = faces.length;

        if (faceCount > 1) {
          noFaceFramesRef.current = 0;
          if (!multiFaceDetectedRef.current) {
            typingIdentityRef.current = null;
            setKeystrokes([]);
          }
          setMultiFaceDetectedSafe(true);
          setFaceDetectedSafe(false);
          faceVectorRef.current = [];
          latestFaceIdentityRef.current = null;
          const l = livenessRef.current;
          if (l.active) {
            failLivenessIdentity("Multiple faces detected. Only one face is allowed.");
          }
          return;
        }

        setMultiFaceDetectedSafe(false);

        if (faceCount === 1) {
          noFaceFramesRef.current = 0;
          const landmarks = faces[0];
          let vector = [];

          // 50 landmarks x (x,y) = 100 features
          for (let i = 0; i < 50; i++) {
            vector.push(landmarks[i].x);
            vector.push(landmarks[i].y);
          }

          faceVectorRef.current = vector;
          const identitySnapshot = buildFaceIdentitySnapshot(landmarks);
          if (identitySnapshot) {
            latestFaceIdentityRef.current = identitySnapshot;
          }
          setFaceDetectedSafe(true);

          const l = livenessRef.current;
          if (l.active) {
            l.missingFrames = 0;
            if (!identitySnapshot || !l.identityRef) {
              failLivenessIdentity("Face tracking lost. Please keep the same person centered.");
              return;
            }

            const similarity = cosineSimilarity(l.identityRef.descriptor, identitySnapshot.descriptor);
            const centerShift = Math.hypot(
              l.identityRef.center.x - identitySnapshot.center.x,
              l.identityRef.center.y - identitySnapshot.center.y
            );
            const areaRatio = identitySnapshot.area / (l.identityRef.area || identitySnapshot.area);
            const identityStable =
              similarity >= IDENTITY_SIMILARITY_MIN &&
              centerShift <= IDENTITY_CENTER_SHIFT_MAX &&
              areaRatio >= IDENTITY_AREA_RATIO_MIN &&
              areaRatio <= IDENTITY_AREA_RATIO_MAX;

            if (identityStable) {
              l.mismatchFrames = Math.max(0, l.mismatchFrames - 1);
              l.identityRef.center.x =
                l.identityRef.center.x * 0.9 + identitySnapshot.center.x * 0.1;
              l.identityRef.center.y =
                l.identityRef.center.y * 0.9 + identitySnapshot.center.y * 0.1;
              l.identityRef.area = l.identityRef.area * 0.9 + identitySnapshot.area * 0.1;
            } else {
              l.mismatchFrames += 1;
              if (l.mismatchFrames >= IDENTITY_MISMATCH_MAX) {
                failLivenessIdentity(
                  "Identity mismatch detected. The same person must stay visible throughout liveness."
                );
                return;
              }
            }
          }

          // Liveness: blink detection with smoothed EAR, both eyes must close
          if (l.active && l.challengeType === "blink" && l.required > 0) {
            try {
              const raw = getEAR(landmarks);
              l.earHistory = (l.earHistory || []).slice(-EAR_SMOOTH_SAMPLES);
              l.earHistory.push(raw.avg);
              const ear =
                l.earHistory.length >= EAR_SMOOTH_SAMPLES
                  ? l.earHistory.reduce((a, b) => a + b, 0) / l.earHistory.length
                  : raw.avg;
              const closedNow =
                ear < EAR_CLOSED ||
                (raw.left < EAR_CLOSED + 0.01 && raw.right < EAR_CLOSED + 0.01);
              const openNow = ear > EAR_OPEN;
              const now = Date.now();

              if (closedNow && !l.eyesClosed) {
                l.eyesClosed = true;
              } else if (
                openNow &&
                l.eyesClosed &&
                now - l.lastBlinkTime > BLINK_COOLDOWN_MS
              ) {
                l.eyesClosed = false;
                l.lastBlinkTime = now;
                l.count += 1;
                setLivenessCountRef.current?.(l.count);
                if (l.count >= l.required) {
                  l.active = false;
                  l.challengeType = null;
                  l.identityRef = null;
                  l.mismatchFrames = 0;
                  l.missingFrames = 0;
                  l.earHistory = [];
                  const done = l.onComplete;
                  l.onComplete = null;
                  if (done) done();
                }
              }
            } catch (e) {
              console.error("Blink detection error:", e);
            }
          }
        } else {
          noFaceFramesRef.current += 1;
          if (noFaceFramesRef.current >= FACE_MISSING_FRAME_TOLERANCE) {
            setFaceDetectedSafe(false);
            faceVectorRef.current = [];
            latestFaceIdentityRef.current = null;
          }
          const l = livenessRef.current;
          if (l.active) {
            l.missingFrames += 1;
            if (l.missingFrames >= IDENTITY_MISSING_MAX) {
              failLivenessIdentity("Face disappeared. Keep your face in frame during liveness.");
            }
          }
        }
      });

      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          const frameInput = getCameraInputFrame();
          if (!frameInput) return;

          await faceMesh.send({ image: frameInput });
          const l = livenessRef.current;
          if (l.active && l.challengeType === "fingers" && handsRef.current) {
            await handsRef.current.send({ image: frameInput });
          }
        },
        width: CAMERA_WIDTH,
        height: CAMERA_HEIGHT,
      });

      camera.start();
      faceMeshRef.current = faceMesh;
    } catch (err) {
      console.error("Camera access denied:", err);
    }
  };

  // ---------- Stop Camera ----------
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    faceVectorRef.current = [];
    latestFaceIdentityRef.current = null;
    typingIdentityRef.current = null;
    noFaceFramesRef.current = 0;
    processingCtxRef.current = null;
    processingCanvasRef.current = null;
    setCameraTuningOpen(false);
    setFaceDetectedSafe(false);
    setMultiFaceDetectedSafe(false);
    setCameraOn(false);
  };

  // ---------- Cleanup camera on unmount ----------
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      faceVectorRef.current = [];
      latestFaceIdentityRef.current = null;
      typingIdentityRef.current = null;
      noFaceFramesRef.current = 0;
      processingCtxRef.current = null;
      processingCanvasRef.current = null;
      setCameraTuningOpen(false);
      faceDetectedRef.current = false;
      multiFaceDetectedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("luma-theme", theme);
  }, [theme]);

  // Pointer + particle interaction layer (no React frame updates)
  useEffect(() => {
    if (typeof window === "undefined") return undefined;

    const palette =
      theme === "light"
        ? [
            [14, 116, 144],
            [79, 70, 229],
            [217, 70, 239],
            [15, 23, 42],
          ]
        : [
            [34, 211, 238],
            [56, 189, 248],
            [167, 139, 250],
            [232, 121, 249],
          ];

    const makeParticle = (width, height) => {
      const color = palette[Math.floor(Math.random() * palette.length)];
      return {
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        size: 1 + Math.random() * 2.4,
        alpha: 0.22 + Math.random() * 0.46,
        drift: 0.3 + Math.random() * 0.7,
        phase: Math.random() * Math.PI * 2,
        color,
      };
    };

    const initializeParticles = () => {
      const { width, height } = pointerMetaRef.current;
      const count = reducedMotionRef.current ? 26 : width < 768 ? 48 : 84;
      particlesRef.current = Array.from({ length: count }, () => makeParticle(width, height));
    };

    const configureCanvas = () => {
      const canvas = particleCanvasRef.current;
      if (!canvas) return;
      const width = window.innerWidth || 1;
      const height = window.innerHeight || 1;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      pointerMetaRef.current = { width, height };
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      particleCtxRef.current = ctx;

      if (pointerTargetRef.current.x === 0 && pointerTargetRef.current.y === 0) {
        pointerTargetRef.current.x = width / 2;
        pointerTargetRef.current.y = height / 2;
        pointerCurrentRef.current.x = width / 2;
        pointerCurrentRef.current.y = height / 2;
      }
    };

    const onReduceMotion = (event) => {
      reducedMotionRef.current = event.matches;
      initializeParticles();
    };

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    reducedMotionRef.current = prefersReducedMotion.matches;

    const onMove = (event) => {
      pointerTargetRef.current.x = event.clientX;
      pointerTargetRef.current.y = event.clientY;
      pointerTargetRef.current.active = true;
    };

    const onLeave = () => {
      pointerTargetRef.current.active = false;
    };

    const onBurst = (event) => {
      const { width, height } = pointerMetaRef.current;
      for (let i = 0; i < 8; i++) {
        const particle = makeParticle(width, height);
        particle.x = event.clientX + (Math.random() - 0.5) * 28;
        particle.y = event.clientY + (Math.random() - 0.5) * 28;
        particle.vx = (Math.random() - 0.5) * 2.1;
        particle.vy = (Math.random() - 0.5) * 2.1;
        particlesRef.current.push(particle);
      }
      if (particlesRef.current.length > 120) {
        particlesRef.current.splice(0, particlesRef.current.length - 120);
      }
    };

    const animate = (timestamp) => {
      const current = pointerCurrentRef.current;
      const target = pointerTargetRef.current;
      const { width, height } = pointerMetaRef.current;
      const dt = Math.min((timestamp - (frameClockRef.current || timestamp)) / 16.67, 2.5);
      frameClockRef.current = timestamp;

      current.x += (target.x - current.x) * 0.12;
      current.y += (target.y - current.y) * 0.12;
      current.active = target.active;

      const ctx = particleCtxRef.current;
      if (ctx) {
        ctx.clearRect(0, 0, width, height);
        const particles = particlesRef.current;
        const attractRadius = reducedMotionRef.current ? 120 : 190;

        for (let i = 0; i < particles.length; i++) {
          const p = particles[i];
          const dx = current.x - p.x;
          const dy = current.y - p.y;
          const dist = Math.hypot(dx, dy) || 1;

          if (current.active && dist < attractRadius) {
            const force = ((attractRadius - dist) / attractRadius) * 0.08 * dt;
            p.vx -= (dx / dist) * force;
            p.vy -= (dy / dist) * force;
          } else if (dist < attractRadius * 1.7) {
            const pull = ((attractRadius * 1.7 - dist) / (attractRadius * 1.7)) * 0.004 * dt;
            p.vx += (dx / dist) * pull;
            p.vy += (dy / dist) * pull;
          }

          p.vx += Math.cos(timestamp * 0.001 + p.phase) * 0.004 * p.drift * dt;
          p.vy += Math.sin(timestamp * 0.0011 + p.phase) * 0.004 * p.drift * dt;
          p.vx *= 0.965;
          p.vy *= 0.965;

          p.x += p.vx * dt;
          p.y += p.vy * dt;

          if (p.x < -20) p.x = width + 20;
          if (p.x > width + 20) p.x = -20;
          if (p.y < -20) p.y = height + 20;
          if (p.y > height + 20) p.y = -20;

          const [r, g, b] = p.color;
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${p.alpha})`;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      pointerAnimRef.current = requestAnimationFrame(animate);
    };

    configureCanvas();
    initializeParticles();
    frameClockRef.current = performance.now();
    pointerAnimRef.current = requestAnimationFrame(animate);

    window.addEventListener("resize", configureCanvas);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerdown", onMove);
    window.addEventListener("pointerdown", onBurst);
    window.addEventListener("pointerleave", onLeave);
    if (prefersReducedMotion.addEventListener) {
      prefersReducedMotion.addEventListener("change", onReduceMotion);
    } else {
      prefersReducedMotion.addListener(onReduceMotion);
    }

    return () => {
      window.removeEventListener("resize", configureCanvas);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerdown", onMove);
      window.removeEventListener("pointerdown", onBurst);
      window.removeEventListener("pointerleave", onLeave);
      if (prefersReducedMotion.removeEventListener) {
        prefersReducedMotion.removeEventListener("change", onReduceMotion);
      } else {
        prefersReducedMotion.removeListener(onReduceMotion);
      }
      if (pointerAnimRef.current) {
        cancelAnimationFrame(pointerAnimRef.current);
      }
    };
  }, [theme]);

  // ---------- Keystroke Capture ----------
  useEffect(() => {
    const handleKey = (e) => {
      // Don't capture keystrokes for shortcuts
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      if (!faceDetectedRef.current || multiFaceDetectedRef.current) return;
      const snapshot = latestFaceIdentityRef.current;
      if (!snapshot) return;
      if (!typingIdentityRef.current) {
        typingIdentityRef.current = cloneIdentitySnapshot(snapshot);
      }
      setKeystrokes((prev) => [...prev, Date.now()]);
    };

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);


  // ---------- Show Toast Notification ----------
  const showToast = (message, type = "success") => {
    setToast({ show: true, message, type });
    setTimeout(() => setToast({ show: false, message: "", type: "success" }), 3000);
  };

  // ---------- Reset Form ----------
  const resetForm = () => {
    setContent("");
    setKeystrokes([]);
    setResult(null);
    typingIdentityRef.current = null;
  };

  // Keep refs in sync for liveness callbacks
  useEffect(() => {
    setLivenessCountRef.current = setLivenessCount;
    setFingersShownRef.current = setFingersShown;
  }, []);

  // ---------- Build Session Vector ----------
  const buildSessionVector = () => {
    const faceVector = faceVectorRef.current;
    let intervals = [];

    if (keystrokes.length > 1) {
      for (let i = 1; i < keystrokes.length; i++) {
        intervals.push((keystrokes[i] - keystrokes[i - 1]) / 1000);
      }
    }

    const mean =
      intervals.length > 0
        ? intervals.reduce((a, b) => a + b, 0) / intervals.length
        : 0;

    const std =
      intervals.length > 0
        ? Math.sqrt(
            intervals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) /
              intervals.length
          )
        : 0;

    const typingVector = [mean, std, intervals.length];

    if (faceVector.length !== 100) {
      showToast("Face not detected properly. Please ensure your face is visible.", "error");
      return null;
    }

    return [...faceVector, ...typingVector]; // 103 features
  };

  // ---------- Run Verify (after liveness) ----------
  const runVerify = async () => {
    const sessionVector = buildSessionVector();
    if (!sessionVector) return;

    setLoading((prev) => ({ ...prev, verify: true }));
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/verify",
        {
          session_vector: sessionVector,
          content: content,
        }
      );

      setResult(response.data);
      showToast("Verification completed successfully!", "success");
    } catch (error) {
      console.error("Backend error:", error);
      setResult(null);
      showToast("Verification failed. Is the backend running at http://127.0.0.1:8000?", "error");
    } finally {
      setLoading((prev) => ({ ...prev, verify: false }));
    }
  };

  const startLiveness = (onComplete) => {
    setCameraTuningOpen(false);

    if (multiFaceDetectedRef.current) {
      showToast("Multiple faces detected. Only one face should be visible.", "error");
      return false;
    }

    const identityAnchor = latestFaceIdentityRef.current || typingIdentityRef.current;
    if (!identityAnchor) {
      showToast("Typing-linked face lock not ready. Type with one visible face first.", "error");
      return false;
    }

    const identityRef = cloneIdentitySnapshot(identityAnchor);

    const rand = Math.random();
    if (rand < 0.5) {
      // Blink challenge: keep short so registration feels immediate.
      const blinkOptions = [1, 1, 1, 2, 2];
      const required = blinkOptions[Math.floor(Math.random() * blinkOptions.length)];
      setLivenessRequired(required);
      setLivenessCount(0);
      setLivenessPrompt(
        `Blink ${required} time${required === 1 ? "" : "s"} and keep the same person who typed in frame`
      );
      setLivenessChallengeType("blink");
      setLivenessActive(true);
      livenessRef.current = {
        active: true,
        required,
        count: 0,
        eyesClosed: false,
        lastBlinkTime: 0,
        onComplete,
        challengeType: "blink",
        fingerCountHistory: [],
        fingerStableSince: null,
        earHistory: [],
        identityRef,
        mismatchFrames: 0,
        missingFrames: 0,
      };
    } else {
      // Finger challenge: 1-5 fingers (weighted toward 2-4)
      const fingerOptions = [1, 2, 2, 3, 3, 3, 4, 4, 5]; // More 2-4s
      const required = fingerOptions[Math.floor(Math.random() * fingerOptions.length)];
      setLivenessRequired(required);
      setFingersShown(0);
      setLivenessCount(0);
      setLivenessPrompt(
        `Show ${required} finger${required === 1 ? "" : "s"} while keeping the same person who typed visible`
      );
      setLivenessChallengeType("fingers");
      setLivenessActive(true);
      livenessRef.current = {
        active: true,
        required,
        count: 0,
        eyesClosed: false,
        lastBlinkTime: 0,
        onComplete,
        challengeType: "fingers",
        fingerCountHistory: [],
        fingerStableSince: null,
        earHistory: [],
        identityRef,
        mismatchFrames: 0,
        missingFrames: 0,
      };
    }
    return true;
  };

  // ---------- Verify (starts liveness first) ----------
  const handleVerify = () => {
    if (multiFaceDetected) {
      showToast("Multiple faces detected. Only one face should be visible.", "error");
      return;
    }
    if (!cameraOn || !faceDetected) {
      showToast("Turn on the camera and ensure your face is detected first.", "error");
      return;
    }
    if (keystrokes.length < 2) {
      showToast("Type at least a few keystrokes before verifying.", "error");
      return;
    }
    if (!validateTypingIdentity()) return;
    if (buildSessionVector() == null) return;

    const started = startLiveness(() => {
      setLivenessActive(false);
      setLivenessCount(0);
      setFingersShown(0);
      runVerify();
    });
    if (!started) return;
  };

  // ---------- Demo: Show bot result (for invigilator / presentation) ----------
  const handleDemoBot = async () => {
    setLoading({ ...loading, demo: true });
    try {
      const vecRes = await axios.get("http://127.0.0.1:8000/demo_bot_vector");
      const sessionVector = vecRes.data.session_vector;
      const response = await axios.post(
        "http://127.0.0.1:8000/verify",
        {
          session_vector: sessionVector,
          content: "demo-bot-example",
        }
      );
      setResult(response.data);
      showToast("Bot demo completed!", "success");
    } catch (error) {
      console.error("Demo bot error:", error);
      setResult(null);
      showToast(
        "Could not run bot demo. Ensure backend is running and you have at least one human sample collected (then train the model).",
        "error"
      );
    } finally {
      setLoading({ ...loading, demo: false });
    }
  };

  // ---------- Run Collect (after liveness) ----------
  const runCollect = async () => {
    const sessionVector = buildSessionVector();
    if (!sessionVector) return;

    setLoading((prev) => ({ ...prev, collect: true }));
    try {
      await axios.post(
        "http://127.0.0.1:8000/collect",
        {
          session_vector: sessionVector,
          content: "training",
          label: 1,
        }
      );

      showToast("Human session stored successfully!", "success");
    } catch (error) {
      console.error("Collect error:", error);
      showToast("Failed to store session. Is the backend running?", "error");
    } finally {
      setLoading((prev) => ({ ...prev, collect: false }));
    }
  };

  // ---------- Collect (starts liveness first) ----------
  const handleCollect = () => {
    if (multiFaceDetected) {
      showToast("Multiple faces detected. Only one face should be visible.", "error");
      return;
    }
    if (!cameraOn || !faceDetected) {
      showToast("Turn on the camera and ensure your face is detected first.", "error");
      return;
    }
    if (keystrokes.length < 2) {
      showToast("Type at least a few keystrokes before collecting.", "error");
      return;
    }
    if (!validateTypingIdentity()) return;
    if (buildSessionVector() == null) return;

    const started = startLiveness(() => {
      setLivenessActive(false);
      setLivenessCount(0);
      setFingersShown(0);
      runCollect();
    });
    if (!started) return;
  };

  const cancelLiveness = () => {
    livenessRef.current.active = false;
    livenessRef.current.onComplete = null;
    livenessRef.current.challengeType = null;
    livenessRef.current.identityRef = null;
    livenessRef.current.mismatchFrames = 0;
    livenessRef.current.missingFrames = 0;
    livenessRef.current.fingerStableSince = null;
    livenessRef.current.fingerCountHistory = [];
    livenessRef.current.earHistory = [];
    setLivenessActive(false);
    setLivenessCount(0);
    setFingersShown(0);
  };

  // ---------- Keyboard Shortcuts ----------
  useEffect(() => {
    const handleShortcut = (e) => {
      // Ctrl/Cmd + Enter: Verify
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        if (!loading.verify && faceDetected && !multiFaceDetected && keystrokes.length >= 2) {
          handleVerify();
        }
      }
      // Ctrl/Cmd + S: Collect
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        if (!loading.collect && faceDetected && !multiFaceDetected && keystrokes.length >= 2) {
          handleCollect();
        }
      }
    };

    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [loading, faceDetected, multiFaceDetected, keystrokes.length]);

  // Calculate typing stats
  const typingStats = keystrokes.length > 1
    ? (() => {
        const intervals = [];
        for (let i = 1; i < keystrokes.length; i++) {
          intervals.push((keystrokes[i] - keystrokes[i - 1]) / 1000);
        }
        const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        const std = Math.sqrt(
          intervals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / intervals.length
        );
        return { mean: mean.toFixed(2), std: std.toFixed(2), count: intervals.length };
      })()
    : null;

  const isLightTheme = theme === "light";
  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  return (
    <div ref={sceneRef} className="experience-shell antigravity-scene min-h-screen relative overflow-hidden">
      <canvas ref={particleCanvasRef} className="particle-canvas" aria-hidden="true" />
      <div className="backdrop-layer app-mesh-bg" />
      <div className="grain-layer" />

      {/* Liveness overlay */}
      {livenessActive && (
        <div className="overlay-backdrop fixed inset-0 z-40 flex items-center justify-center p-4 animate-fade-in">
          <div className="liveness-modal rounded-3xl p-8 sm:p-10 max-w-md w-full animate-fade-in-scale shadow-2xl">
            <div className="challenge-badge animate-liveness-pulse">
              <span>{livenessChallengeType === "blink" ? "BLINK" : "HAND"}</span>
            </div>
            <h3 className="modal-title">Identity Liveness Check</h3>
            <p className="modal-copy">Complete this as the same person who typed in the box.</p>
            {livenessChallengeType === "fingers" && (
              <div className="challenge-number">{livenessRequired}</div>
            )}
            <p className="challenge-prompt">{livenessPrompt}</p>
            {livenessChallengeType === "blink" && (
              <div className="challenge-block">
                <div className="challenge-dots">
                  {Array.from({ length: livenessRequired }).map((_, i) => (
                    <span
                      key={i}
                      className={`challenge-dot ${i < livenessCount ? "challenge-dot-active" : ""}`}
                    />
                  ))}
                </div>
                <div className="challenge-value">{livenessCount} / {livenessRequired}</div>
                <p className="challenge-meta">Blinks detected</p>
                {livenessCount < livenessRequired && (
                  <p className="challenge-hint">Blink naturally: close then open your eyes.</p>
                )}
              </div>
            )}
            {livenessChallengeType === "fingers" && (
              <div className="challenge-block">
                <div className="finger-bars">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <span key={i} className={`finger-bar ${i < fingersShown ? "finger-bar-active" : ""}`} />
                  ))}
                </div>
                <div className="challenge-value">{fingersShown} / {livenessRequired}</div>
                <p className="challenge-meta">Fingers shown</p>
                {fingersShown === livenessRequired && <p className="challenge-hint success">Hold steady...</p>}
                {fingersShown !== livenessRequired && fingersShown > 0 && (
                  <p className="challenge-hint">Adjust to show {livenessRequired} finger{livenessRequired === 1 ? "" : "s"}.</p>
                )}
              </div>
            )}
            <button type="button" onClick={cancelLiveness} className="cancel-button">
              Cancel
            </button>
          </div>
        </div>
      )}

      {cameraTuningOpen && (
        <div className="camera-popup-backdrop" onClick={closeCameraTuning}>
          <div className="camera-popup" onClick={(e) => e.stopPropagation()}>
            <div className="camera-popup-head">
              <p className="camera-popup-title">Light Adjustment</p>
              <button type="button" className="camera-popup-close" onClick={closeCameraTuning}>
                Close
              </button>
            </div>

            <div className="camera-tuning-panel">
              <label className="camera-tuning-row">
                <span>Light {cameraFilter.brightness}%</span>
                <input
                  type="range"
                  min={CAMERA_BRIGHTNESS_MIN}
                  max={CAMERA_BRIGHTNESS_MAX}
                  step={1}
                  value={cameraFilter.brightness}
                  onChange={(e) => updateCameraFilter("brightness", parseInt(e.target.value, 10))}
                />
              </label>

              <label className="camera-tuning-row">
                <span>Contrast {cameraFilter.contrast}%</span>
                <input
                  type="range"
                  min={CAMERA_CONTRAST_MIN}
                  max={CAMERA_CONTRAST_MAX}
                  step={1}
                  value={cameraFilter.contrast}
                  onChange={(e) => updateCameraFilter("contrast", parseInt(e.target.value, 10))}
                />
              </label>
            </div>

            <div className="camera-popup-actions">
              <button type="button" onClick={resetCameraFilter} className="camera-tuning-reset">
                Reset
              </button>
            </div>
          </div>
        </div>
      )}

      {toast.show && (
        <div className={`toast-chip animate-slide-in ${toast.type === "success" ? "toast-success" : "toast-error"}`}>
          <span className="toast-symbol">{toast.type === "success" ? "OK" : "ERR"}</span>
          <span className="toast-message">{toast.message}</span>
        </div>
      )}

      <div className="content-shell">
        <header className="topbar animate-fade-in">
          <div className="brand-cluster">
            <div className="brand-mark">LX</div>
            <div>
              <p className="brand-caption">Realtime Human Verification Platform</p>
              <h1 className="brand-title">LUMA-X</h1>
            </div>
          </div>
          <button
            type="button"
            onClick={toggleTheme}
            className="theme-toggle"
            aria-label={`Switch to ${isLightTheme ? "dark" : "light"} mode`}
          >
            {isLightTheme ? "Switch to dark" : "Switch to light"}
          </button>
        </header>

        <main className="dashboard-grid">
          <section className="main-panel capture-panel animate-fade-in">
            <div className="section-head">
              <h2>Live Capture</h2>
              <p>Single-subject feed with anti-handoff identity continuity.</p>
            </div>

            <div className="camera-shell">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="video-frame"
                style={{
                  filter: `brightness(${cameraFilter.brightness}%) contrast(${cameraFilter.contrast}%)`,
                }}
              />
              {cameraOn && (
                <div className="status-pill">
                  <span
                    className={`status-dot ${
                      multiFaceDetected
                        ? "status-alert"
                        : faceDetected
                          ? "status-ok"
                          : "status-off"
                    }`}
                  />
                  <span className="status-label">
                    {multiFaceDetected ? "Multiple faces" : faceDetected ? "Face detected" : "No face"}
                  </span>
                </div>
              )}
            </div>

            <div className="camera-actions">
              {!cameraOn ? (
                <button onClick={startCamera} className="pill-btn pill-start">
                  Start camera
                </button>
              ) : (
                <button onClick={stopCamera} className="pill-btn pill-stop">
                  Stop camera
                </button>
              )}
              <button
                type="button"
                onClick={toggleCameraTuning}
                className="pill-btn pill-neutral"
                disabled={!cameraOn}
              >
                {cameraTuningOpen ? "Close light controls" : "Adjust light"}
              </button>
            </div>

            {multiFaceDetected && (
              <p className="hint-warning">Only one face is allowed for verification.</p>
            )}

            {typingStats && (
              <div className="stats-strip animate-fade-in">
                <div className="stat-card">
                  <p className="stat-label">Keystrokes</p>
                  <p className="stat-value">{keystrokes.length}</p>
                </div>
                <div className="stat-card">
                  <p className="stat-label">Mean Delay</p>
                  <p className="stat-value">{typingStats.mean}s</p>
                </div>
                <div className="stat-card">
                  <p className="stat-label">Std Dev</p>
                  <p className="stat-value">{typingStats.std}s</p>
                </div>
              </div>
            )}
          </section>

          <section className="main-panel flow-panel animate-fade-in">
            <div className="section-head">
              <h2>Verification Flow</h2>
              <p>Type naturally, then complete liveness in one uninterrupted sequence.</p>
            </div>

            <label htmlFor="typing-input" className="field-label">Behavior Sample</label>
            <textarea
              id="typing-input"
              className="typing-input"
              rows="5"
              placeholder="Type a natural paragraph here to capture timing patterns..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
            />
            {keystrokes.length === 0 && <p className="field-hint">Start typing to capture keystroke dynamics.</p>}

            <div className="actions-stack">
              <button
                onClick={handleVerify}
                disabled={loading.verify || !faceDetected || multiFaceDetected || keystrokes.length < 2}
                className="cta cta-primary"
              >
                {loading.verify ? (
                  <>
                    <span className="inline-spinner" />
                    Verifying...
                  </>
                ) : (
                  <>
                    <span>Verify human presence</span>
                    <span className="shortcut-chip">Ctrl+Enter</span>
                  </>
                )}
              </button>

              <button
                onClick={handleCollect}
                disabled={loading.collect || !faceDetected || multiFaceDetected || keystrokes.length < 2}
                className="cta cta-secondary"
              >
                {loading.collect ? (
                  <>
                    <span className="inline-spinner" />
                    Collecting...
                  </>
                ) : (
                  <>
                    <span>Collect human sample</span>
                    <span className="shortcut-chip">Ctrl+S</span>
                  </>
                )}
              </button>

              <button onClick={handleDemoBot} disabled={loading.demo} className="cta cta-tertiary">
                {loading.demo ? (
                  <>
                    <span className="inline-spinner" />
                    Running demo...
                  </>
                ) : (
                  "Demo: show bot result"
                )}
              </button>

              <button type="button" onClick={resetForm} className="cta cta-ghost">
                Refresh session
              </button>
            </div>

            {result && (
              <div className="result-panel animate-result-success">
                {typeof result.human_probability === "number" && (
                  <div className="probability-block">
                    <div className="probability-header">
                      <span>Calibrated human confidence</span>
                      <strong>{(result.human_probability * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="probability-track">
                      <div
                        className={`probability-fill ${result.prediction === 1 ? "probability-human" : "probability-bot"}`}
                        style={{ width: `${result.human_probability * 100}%` }}
                      />
                    </div>
                    <p className="confidence-note">
                      Confidence score only. Dataset accuracy is measured during training, not per request.
                    </p>
                  </div>
                )}

                <div className="prediction-block">
                  <p className="prediction-label">Prediction</p>
                  <p className={`prediction-value ${result.prediction === 1 ? "prediction-human" : "prediction-bot"}`}>
                    {result.prediction === 1 ? "Human verified" : "Bot detected"}
                  </p>
                </div>

                {result.hash && (
                  <div className="hash-section">
                    <p className="hash-label">Verification hash</p>
                    <div className="hash-box">{result.hash}</div>
                  </div>
                )}
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}

export default App;

