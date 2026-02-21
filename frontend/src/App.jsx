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

const EAR_CLOSED = 0.20;   // Eyes considered closed
const EAR_OPEN = 0.28;     // Eyes considered open (higher = stricter)
const BLINK_COOLDOWN_MS = 350;  // Min ms between blinks
const EAR_SMOOTH_SAMPLES = 7;   // Running average for stability
const FINGER_SMOOTH_SAMPLES = 5; // Majority vote for finger count
const FINGER_STABLE_MS = 500;   // Must hold correct count this long
const IDENTITY_SIMILARITY_MIN = 0.94;
const IDENTITY_CENTER_SHIFT_MAX = 0.17;
const IDENTITY_AREA_RATIO_MIN = 0.58;
const IDENTITY_AREA_RATIO_MAX = 1.72;
const IDENTITY_MISMATCH_MAX = 6;
const IDENTITY_MISSING_MAX = 8;
const FACE_SIGNATURE_POINTS = [33, 263, 1, 61, 291, 4, 10, 152, 70, 300, 234, 454];

// Count extended fingers from MediaPipe Hands (21 landmarks). Robust algorithm.
function countExtendedFingers(landmarks) {
  if (!landmarks || landmarks.length < 21) return 0;
  let count = 0;
  const wrist = landmarks[0];

  const dist = (a, b) =>
    Math.hypot(landmarks[a].x - landmarks[b].x, landmarks[a].y - landmarks[b].y);

  // Four fingers: tip farther from wrist than MCP = extended
  const fingers = [
    { tip: 8, mcp: 5 },   // Index
    { tip: 12, mcp: 9 },  // Middle
    { tip: 16, mcp: 13 }, // Ring
    { tip: 20, mcp: 17 }, // Pinky
  ];
  for (const { tip, mcp } of fingers) {
    const dTip = dist(tip, 0);
    const dMcp = dist(mcp, 0);
    if (dTip > dMcp * 1.05) count++; // Tip extends beyond MCP
  }

  // Thumb: tip (4) farther from wrist than IP (3)
  const dThumbTip = dist(4, 0);
  const dThumbIp = dist(3, 0);
  if (dThumbTip > dThumbIp * 1.08) count++;

  return Math.min(5, Math.max(0, count));
}

// Majority vote from recent counts for stability
function smoothFingerCount(history, required) {
  if (history.length < FINGER_SMOOTH_SAMPLES) return null;
  const recent = history.slice(-FINGER_SMOOTH_SAMPLES);
  const counts = {};
  for (const c of recent) {
    counts[c] = (counts[c] || 0) + 1;
  }
  let best = -1, bestCount = 0;
  for (const [c, n] of Object.entries(counts)) {
    if (n > bestCount) {
      bestCount = n;
      best = parseInt(c, 10);
    }
  }
  return best === required ? best : null;
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

function App() {
  const [content, setContent] = useState("");
  const [keystrokes, setKeystrokes] = useState([]);
  const [cameraOn, setCameraOn] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState({ verify: false, collect: false, demo: false });
  const [toast, setToast] = useState({ show: false, message: "", type: "success" });
  const [faceDetected, setFaceDetected] = useState(false);
  const [multiFaceDetected, setMultiFaceDetected] = useState(false);
  const faceVectorRef = useRef([]);
  const latestFaceIdentityRef = useRef(null);
  const faceDetectedRef = useRef(false);
  const multiFaceDetectedRef = useRef(false);
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "dark";
    const saved = window.localStorage.getItem("luma-theme");
    if (saved === "dark" || saved === "light") return saved;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  });
  const sceneRef = useRef(null);
  const cardRef = useRef(null);
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

  // ---------- Start Camera ----------
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      setCameraOn(true);

      const faceMesh = new FaceMesh({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 2,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
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
          const count = countExtendedFingers(results.multiHandLandmarks[0]);
          l.fingerCountHistory = (l.fingerCountHistory || []).slice(-FINGER_SMOOTH_SAMPLES);
          l.fingerCountHistory.push(count);
          setFingersShownRef.current?.(count);

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
              const bothClosed = raw.left < EAR_CLOSED && raw.right < EAR_CLOSED;
              const bothOpen = raw.left >= EAR_OPEN && raw.right >= EAR_OPEN;
              const now = Date.now();

              if (bothClosed && !l.eyesClosed) {
                l.eyesClosed = true;
              } else if (
                bothOpen &&
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
          setFaceDetectedSafe(false);
          faceVectorRef.current = [];
          latestFaceIdentityRef.current = null;
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
          await faceMesh.send({ image: videoRef.current });
          const l = livenessRef.current;
          if (l.active && l.challengeType === "fingers" && handsRef.current) {
            await handsRef.current.send({ image: videoRef.current });
          }
        },
        width: 640,
        height: 480,
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

      const normX = ((current.x / width) - 0.5) * 2;
      const normY = ((current.y / height) - 0.5) * 2;

      if (cardRef.current) {
        cardRef.current.style.setProperty("--tilt-x", `${(-normY * 4.5).toFixed(2)}`);
        cardRef.current.style.setProperty("--tilt-y", `${(normX * 6.5).toFixed(2)}`);
      }

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
    if (multiFaceDetectedRef.current) {
      showToast("Multiple faces detected. Only one face should be visible.", "error");
      return false;
    }

    const identityAnchor = latestFaceIdentityRef.current;
    if (!identityAnchor) {
      showToast("Face lock not ready. Keep your face centered and try again.", "error");
      return false;
    }

    const identityRef = {
      descriptor: [...identityAnchor.descriptor],
      center: {
        x: identityAnchor.center.x,
        y: identityAnchor.center.y,
      },
      area: identityAnchor.area,
    };

    const rand = Math.random();
    if (rand < 0.5) {
      // Blink challenge: 1-3 blinks (weighted toward 2)
      const blinkOptions = [1, 2, 2, 2, 3]; // More 2s for common case
      const required = blinkOptions[Math.floor(Math.random() * blinkOptions.length)];
      setLivenessRequired(required);
      setLivenessCount(0);
      setLivenessPrompt(
        `Blink ${required} time${required === 1 ? "" : "s"} and keep the same face in frame`
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
        `Show ${required} finger${required === 1 ? "" : "s"} while keeping the same face visible`
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
    <div ref={sceneRef} className="app-shell antigravity-scene min-h-screen flex flex-col items-center justify-center p-4 sm:p-6 relative overflow-hidden">
      <canvas ref={particleCanvasRef} className="particle-canvas" aria-hidden="true" />
      <div className="absolute inset-0 app-mesh-bg" />
      <button
        type="button"
        onClick={toggleTheme}
        className="theme-toggle z-20"
        aria-label={`Switch to ${isLightTheme ? "dark" : "light"} mode`}
      >
        <span className="font-medium">{isLightTheme ? "Dark mode" : "Light mode"}</span>
      </button>

      {/* Liveness overlay */}
      {livenessActive && (
        <div className="fixed inset-0 z-40 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xl animate-fade-in">
          <div className="liveness-modal backdrop-blur-2xl border border-white/10 rounded-3xl p-8 sm:p-10 max-w-md w-full text-center animate-fade-in-scale shadow-2xl">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-cyan-500/15 border border-cyan-500/30 mb-6 animate-liveness-pulse">
              <span className="text-sm font-semibold">{livenessChallengeType === "blink" ? "BLINK" : "HAND"}</span>
            </div>
            <h3 className="font-display text-2xl font-bold text-white mb-1">Liveness check</h3>
            <p className="text-white/60 text-sm mb-6">Prove you are live - not a photo or screen</p>
            {livenessChallengeType === "fingers" && (
              <div className="text-7xl font-extrabold bg-gradient-to-r from-cyan-400 via-violet-400 to-fuchsia-400 bg-clip-text text-transparent my-6">
                {livenessRequired}
              </div>
            )}
            <p className="text-cyan-400 font-semibold mb-6">{livenessPrompt}</p>
            {livenessChallengeType === "blink" && (
              <div className="mb-6">
                <div className="flex items-center justify-center gap-2 mb-2">
                  {Array.from({ length: livenessRequired }).map((_, i) => (
                    <div
                      key={i}
                      className={`w-3 h-3 rounded-full transition-all duration-300 ${i < livenessCount ? "bg-cyan-400 scale-110 shadow-[0_0_20px_rgba(34,211,238,0.5)]" : "bg-white/20"}`}
                    />
                  ))}
                </div>
                <div className="text-2xl font-bold text-white">{livenessCount} / {livenessRequired}</div>
                <div className="text-sm text-white/50 mt-1">Blinks detected</div>
                {livenessCount < livenessRequired && (
                  <p className="mt-3 text-xs text-white/40">Blink naturally - close then open your eyes</p>
                )}
              </div>
            )}
            {livenessChallengeType === "fingers" && (
              <div className="mb-6">
                <div className="flex items-center justify-center gap-2 mb-2">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div
                      key={i}
                      className={`w-2 h-8 rounded-full transition-all duration-300 ${i < fingersShown ? "bg-amber-400/90" : "bg-white/15"}`}
                    />
                  ))}
                </div>
                <div className="text-2xl font-bold text-white">{fingersShown} / {livenessRequired}</div>
                <div className="text-sm text-white/50 mt-1">Fingers shown</div>
                {fingersShown === livenessRequired && (
                  <p className="mt-3 text-xs text-emerald-400 animate-pulse">Hold steady...</p>
                )}
                {fingersShown !== livenessRequired && fingersShown > 0 && (
                  <p className="mt-3 text-xs text-amber-400/90">Adjust to show {livenessRequired} finger{livenessRequired === 1 ? "" : "s"}</p>
                )}
              </div>
            )}
            <button
              type="button"
              onClick={cancelLiveness}
              className="cancel-button px-5 py-2.5 rounded-xl border border-white/20 text-white/70 transition-all duration-200"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {toast.show && (
        <div
          className={`fixed top-6 right-6 z-50 px-5 py-3 rounded-2xl shadow-2xl backdrop-blur-xl border animate-slide-in flex items-center gap-3 ${
            toast.type === "success" ? "bg-emerald-500/90 border-emerald-400/50 text-white" : "bg-red-500/90 border-red-400/50 text-white"
          }`}
        >
          <span className="text-lg">{toast.type === "success" ? "OK" : "ERR"}</span>
          <span className="font-medium">{toast.message}</span>
        </div>
      )}

      <div ref={cardRef} className="main-panel reactive-card relative w-full max-w-2xl backdrop-blur-2xl rounded-3xl p-6 sm:p-10 border shadow-2xl animate-fade-in transition-transform duration-300 ease-out will-change-transform">
        <div className="text-center mb-8">
          <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-extrabold mb-3 bg-gradient-to-r from-cyan-400 via-violet-400 to-fuchsia-400 bg-clip-text text-transparent tracking-tight">
            LUMA-X
          </h1>
          <p className="text-white/50 text-sm sm:text-base">Multi-Modal Human Presence Attestation</p>
        </div>

        <div className="mb-6 flex flex-col items-center">
          <div className="relative group">
            <div className={`absolute -inset-1 rounded-2xl transition-all duration-500 ${faceDetected ? "bg-cyan-500/20 blur-sm" : "bg-white/5"}`} />
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="relative rounded-2xl border border-white/10 w-full max-w-xs aspect-video object-cover shadow-2xl"
            />
            {cameraOn && (
              <div className="status-pill absolute top-3 right-3 flex items-center gap-2 px-3 py-1.5 rounded-full backdrop-blur-sm border border-white/10">
                <div
                  className={`w-2.5 h-2.5 rounded-full ${
                    multiFaceDetected
                      ? "bg-amber-400 shadow-[0_0_12px_rgba(251,191,36,0.6)]"
                      : faceDetected
                        ? "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.6)] animate-pulse"
                        : "bg-red-400"
                  }`}
                />
                <span className="text-xs font-medium text-white/90">
                  {multiFaceDetected ? "Multiple faces" : faceDetected ? "Face detected" : "No face"}
                </span>
              </div>
            )}
          </div>
          <div className="flex gap-3 mt-5">
            {!cameraOn ? (
              <button
                onClick={startCamera}
                className="px-6 py-3 rounded-xl font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 hover:bg-emerald-500/30 hover:scale-[1.02] hover:-translate-y-0.5 active:scale-[0.98] transition-all duration-200"
              >
                Start camera
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="px-6 py-3 rounded-xl font-semibold bg-red-500/20 text-red-400 border border-red-500/40 hover:bg-red-500/30 hover:scale-[1.02] hover:-translate-y-0.5 active:scale-[0.98] transition-all duration-200"
              >
                Stop camera
              </button>
            )}
          </div>
          {multiFaceDetected && (
            <p className="mt-3 text-xs text-amber-400">Only one face is allowed for verification.</p>
          )}
        </div>

        {typingStats && (
          <div className="stats-panel mb-5 p-4 rounded-2xl border animate-fade-in">
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-xs text-white/40 font-medium uppercase tracking-wider">Keystrokes</div>
                <div className="text-xl font-bold text-cyan-400">{keystrokes.length}</div>
              </div>
              <div>
                <div className="text-xs text-white/40 font-medium uppercase tracking-wider">Mean</div>
                <div className="text-xl font-bold text-violet-400">{typingStats.mean}s</div>
              </div>
              <div>
                <div className="text-xs text-white/40 font-medium uppercase tracking-wider">Std</div>
                <div className="text-xl font-bold text-fuchsia-400">{typingStats.std}s</div>
              </div>
            </div>
          </div>
        )}

        <textarea
          className="typing-input w-full p-4 rounded-2xl border placeholder:text-white/30 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/30 transition-all duration-200 resize-none text-white"
          rows="4"
          placeholder="Type here to capture keystroke patterns..."
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />
        {keystrokes.length === 0 && <p className="mt-2 text-xs text-white/30 text-center">Start typing to capture data</p>}

        <div className="mt-6 space-y-3">
          <button
            onClick={handleVerify}
            disabled={loading.verify || !faceDetected || multiFaceDetected || keystrokes.length < 2}
            className="w-full py-4 rounded-2xl font-semibold flex items-center justify-center gap-3 bg-cyan-500/15 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/25 hover:shadow-[0_0_30px_-5px_rgba(34,211,238,0.4)] hover:scale-[1.02] hover:-translate-y-0.5 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:translate-y-0 transition-all duration-200"
          >
            {loading.verify ? (
              <>
                <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                Verifying...
              </>
            ) : (
              <>
                <span>Verify human presence</span>
                <span className="text-xs opacity-60">Ctrl+Enter</span>
              </>
            )}
          </button>
          <button
            onClick={handleCollect}
            disabled={loading.collect || !faceDetected || multiFaceDetected || keystrokes.length < 2}
            className="w-full py-4 rounded-2xl font-semibold flex items-center justify-center gap-3 bg-violet-500/15 text-violet-400 border border-violet-500/30 hover:bg-violet-500/25 hover:shadow-[0_0_30px_-5px_rgba(167,139,250,0.4)] hover:scale-[1.02] hover:-translate-y-0.5 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:translate-y-0 transition-all duration-200"
          >
            {loading.collect ? (
              <>
                <div className="w-5 h-5 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                Collecting...
              </>
            ) : (
              <>
                <span>Collect human sample</span>
                <span className="text-xs opacity-60">Ctrl+S</span>
              </>
            )}
          </button>
          <button
            onClick={handleDemoBot}
            disabled={loading.demo}
            className="w-full py-4 rounded-2xl font-semibold flex items-center justify-center gap-3 bg-amber-500/15 text-amber-400 border border-amber-500/30 hover:bg-amber-500/25 hover:scale-[1.02] hover:-translate-y-0.5 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:translate-y-0 transition-all duration-200"
          >
            {loading.demo ? (
              <div className="w-5 h-5 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
            ) : (
              "Demo: show bot result"
            )}
          </button>
          <button
            type="button"
            onClick={resetForm}
            className="neutral-button w-full py-3 rounded-2xl font-medium border text-white/50 transition-all duration-200"
          >
            Refresh
          </button>
        </div>

        {result && (
          <div className="result-panel mt-8 p-6 rounded-2xl border animate-result-success shadow-xl">
            {typeof result.human_probability === "number" && (
              <div className="mb-5">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-white/50 font-medium">Human probability</span>
                  <span className="text-2xl font-bold text-cyan-400">
                    {(result.human_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ease-out ${
                      result.prediction === 1
                        ? "bg-gradient-to-r from-emerald-500 to-emerald-400"
                        : "bg-gradient-to-r from-red-500 to-red-400"
                    }`}
                    style={{ width: `${result.human_probability * 100}%` }}
                  />
                </div>
              </div>
            )}
            <div className="text-center py-4">
              <div className="text-xs text-white/40 uppercase tracking-widest mb-2">Prediction</div>
              <div className={`text-2xl font-bold ${result.prediction === 1 ? "text-emerald-400" : "text-red-400"}`}>
                {result.prediction === 1 ? "Human verified" : "Bot detected"}
              </div>
            </div>
            {result.hash && (
              <div className="pt-4 border-t border-white/10">
                <div className="text-xs text-white/40 uppercase tracking-wider mb-1">Verification hash</div>
                <div className="hash-box text-xs font-mono break-all text-white/50 p-3 rounded-xl">
                  {result.hash}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

