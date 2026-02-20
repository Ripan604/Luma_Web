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

function App() {
  const [content, setContent] = useState("");
  const [keystrokes, setKeystrokes] = useState([]);
  const [cameraOn, setCameraOn] = useState(false);
  const [result, setResult] = useState(null);
  const [faceVector, setFaceVector] = useState([]);
  const [loading, setLoading] = useState({ verify: false, collect: false, demo: false });
  const [toast, setToast] = useState({ show: false, message: "", type: "success" });
  const [faceDetected, setFaceDetected] = useState(false);

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
  });
  const setLivenessCountRef = useRef(() => {});

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const faceMeshRef = useRef(null);
  const handsRef = useRef(null);
  const [fingersShown, setFingersShown] = useState(0);
  const setFingersShownRef = useRef(() => {});

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
        maxNumFaces: 1,
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
        if (results.multiFaceLandmarks?.length > 0) {
          const landmarks = results.multiFaceLandmarks[0];
          let vector = [];

          // 50 landmarks × (x,y) = 100 features
          for (let i = 0; i < 50; i++) {
            vector.push(landmarks[i].x);
            vector.push(landmarks[i].y);
          }

          setFaceVector(vector);
          setFaceDetected(true);

          // Liveness: blink detection with smoothed EAR, both eyes must close
          const l = livenessRef.current;
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
          setFaceDetected(false);
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
    };
  }, []);

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
    const rand = Math.random();
    if (rand < 0.5) {
      // Blink challenge: 1-3 blinks (weighted toward 2)
      const blinkOptions = [1, 2, 2, 2, 3]; // More 2s for common case
      const required = blinkOptions[Math.floor(Math.random() * blinkOptions.length)];
      setLivenessRequired(required);
      setLivenessCount(0);
      setLivenessPrompt(`Blink ${required} time${required === 1 ? "" : "s"} to prove you're live (not a photo)`);
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
      };
    } else {
      // Finger challenge: 1-5 fingers (weighted toward 2-4)
      const fingerOptions = [1, 2, 2, 3, 3, 3, 4, 4, 5]; // More 2-4s
      const required = fingerOptions[Math.floor(Math.random() * fingerOptions.length)];
      setLivenessRequired(required);
      setFingersShown(0);
      setLivenessCount(0);
      setLivenessPrompt(`Show ${required} finger${required === 1 ? "" : "s"} to the camera`);
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
      };
    }
  };

  // ---------- Verify (starts liveness first) ----------
  const handleVerify = () => {
    if (!cameraOn || !faceDetected) {
      showToast("Turn on the camera and ensure your face is detected first.", "error");
      return;
    }
    if (keystrokes.length < 2) {
      showToast("Type at least a few keystrokes before verifying.", "error");
      return;
    }
    if (buildSessionVector() == null) return;

    startLiveness(() => {
      setLivenessActive(false);
      setLivenessCount(0);
      setFingersShown(0);
      runVerify();
    });
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

      showToast("Human session stored successfully! ✅", "success");
    } catch (error) {
      console.error("Collect error:", error);
      showToast("Failed to store session. Is the backend running?", "error");
    } finally {
      setLoading((prev) => ({ ...prev, collect: false }));
    }
  };

  // ---------- Collect (starts liveness first) ----------
  const handleCollect = () => {
    if (!cameraOn || !faceDetected) {
      showToast("Turn on the camera and ensure your face is detected first.", "error");
      return;
    }
    if (keystrokes.length < 2) {
      showToast("Type at least a few keystrokes before collecting.", "error");
      return;
    }
    if (buildSessionVector() == null) return;

    startLiveness(() => {
      setLivenessActive(false);
      setLivenessCount(0);
      setFingersShown(0);
      runCollect();
    });
  };

  const cancelLiveness = () => {
    livenessRef.current.active = false;
    livenessRef.current.onComplete = null;
    livenessRef.current.challengeType = null;
    livenessRef.current.fingerStableSince = null;
    livenessRef.current.fingerCountHistory = [];
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
        if (!loading.verify && faceDetected && keystrokes.length >= 2) {
          handleVerify();
        }
      }
      // Ctrl/Cmd + S: Collect
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        if (!loading.collect && faceDetected && keystrokes.length >= 2) {
          handleCollect();
        }
      }
    };

    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [loading, faceDetected, keystrokes.length]);

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

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white flex flex-col items-center justify-center p-4 sm:p-6 relative overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(34,211,238,0.15),transparent)]" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_40%_at_80%_100%,rgba(167,139,250,0.12),transparent)]" />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-float" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-violet-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: "-3s" }} />

      {/* Liveness overlay */}
      {livenessActive && (
        <div className="fixed inset-0 z-40 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xl animate-fade-in">
          <div className="bg-[#16161f]/95 backdrop-blur-2xl border border-white/10 rounded-3xl p-8 sm:p-10 max-w-md w-full text-center animate-fade-in-scale shadow-2xl">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-cyan-500/15 border border-cyan-500/30 mb-6 animate-liveness-pulse">
              <span className="text-2xl">{livenessChallengeType === "blink" ? "👁" : "✋"}</span>
            </div>
            <h3 className="font-display text-2xl font-bold text-white mb-1">Liveness check</h3>
            <p className="text-white/60 text-sm mb-6">Prove you are live — not a photo or screen</p>
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
                  <p className="mt-3 text-xs text-white/40">Blink naturally — close then open your eyes</p>
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
              className="px-5 py-2.5 rounded-xl border border-white/20 text-white/70 hover:bg-white/10 hover:text-white transition-all duration-200"
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
          <span className="text-lg">{toast.type === "success" ? "✓" : "✗"}</span>
          <span className="font-medium">{toast.message}</span>
        </div>
      )}

      <div className="relative w-full max-w-2xl bg-[#16161f]/80 backdrop-blur-2xl rounded-3xl p-6 sm:p-10 border border-white/[0.06] shadow-2xl animate-fade-in">
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
              <div className="absolute top-3 right-3 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-sm border border-white/10">
                <div className={`w-2.5 h-2.5 rounded-full ${faceDetected ? "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.6)] animate-pulse" : "bg-red-400"}`} />
                <span className="text-xs font-medium text-white/90">{faceDetected ? "Face detected" : "No face"}</span>
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
        </div>

        {typingStats && (
          <div className="mb-5 p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06] animate-fade-in">
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
          className="w-full p-4 rounded-2xl bg-white/[0.04] border border-white/10 placeholder:text-white/30 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/30 transition-all duration-200 resize-none text-white"
          rows="4"
          placeholder="Type here to capture keystroke patterns..."
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />
        {keystrokes.length === 0 && <p className="mt-2 text-xs text-white/30 text-center">Start typing to capture data</p>}

        <div className="mt-6 space-y-3">
          <button
            onClick={handleVerify}
            disabled={loading.verify || !faceDetected || keystrokes.length < 2}
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
            disabled={loading.collect || !faceDetected || keystrokes.length < 2}
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
            className="w-full py-3 rounded-2xl font-medium border border-white/15 text-white/50 hover:bg-white/5 hover:text-white/70 transition-all duration-200"
          >
            Refresh
          </button>
        </div>

        {result && (
          <div className="mt-8 p-6 rounded-2xl bg-white/[0.04] border border-white/10 animate-result-success shadow-xl" style={{ boxShadow: "0 0 0 1px rgba(255,255,255,0.05) inset" }}>
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
                <div className="text-xs font-mono break-all text-white/50 bg-black/30 p-3 rounded-xl">
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
