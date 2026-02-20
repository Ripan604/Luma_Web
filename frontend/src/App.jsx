import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { FaceMesh } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";
import { Hands } from "@mediapipe/hands";

// Eye Aspect Ratio (EAR): lower = more closed. Standard MediaPipe Face Mesh indices.
function getEAR(landmarks) {
  const d = (a, b) =>
    Math.hypot(landmarks[a].x - landmarks[b].x, landmarks[a].y - landmarks[b].y);
  
  // Left eye: vertical pairs (159-145, 158-153), horizontal (33-133)
  const leftVert1 = d(159, 145);
  const leftVert2 = d(158, 153);
  const leftHoriz = d(33, 133) || 0.001;
  const earLeft = (leftVert1 + leftVert2) / (2 * leftHoriz);
  
  // Right eye: vertical pairs (386-374, 385-380), horizontal (362-263)
  const rightVert1 = d(386, 374);
  const rightVert2 = d(385, 380);
  const rightHoriz = d(362, 263) || 0.001;
  const earRight = (rightVert1 + rightVert2) / (2 * rightHoriz);
  
  return (earLeft + earRight) / 2;
}

const EAR_CLOSED = 0.22;  // More lenient threshold
const EAR_OPEN = 0.28;    // Higher threshold to ensure eyes fully open
const BLINK_COOLDOWN_MS = 300;  // Shorter cooldown for faster detection

// Count extended fingers from MediaPipe Hands (21 landmarks). Improved algorithm.
function countExtendedFingers(landmarks) {
  if (!landmarks || landmarks.length < 21) return 0;
  let count = 0;
  
  // Four fingers: tip must be above PIP (lower y = higher on screen)
  // Also check tip is above MCP for more accuracy
  const fingerChecks = [
    { tip: 8, pip: 6, mcp: 5 },   // Index
    { tip: 12, pip: 10, mcp: 9 }, // Middle
    { tip: 16, pip: 14, mcp: 13 }, // Ring
    { tip: 20, pip: 18, mcp: 17 }, // Pinky
  ];
  
  for (const { tip, pip, mcp } of fingerChecks) {
    const tipY = landmarks[tip].y;
    const pipY = landmarks[pip].y;
    const mcpY = landmarks[mcp].y;
    // Extended if tip is above both PIP and MCP
    if (tipY < pipY && tipY < mcpY) {
      count++;
    }
  }
  
  // Thumb: check if tip (4) is extended outward from hand
  // Compare tip to IP (3) and MCP (2) - thumb extended if tip is farther from wrist
  const wrist = landmarks[0];
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const thumbMcp = landmarks[2];
  
  const dTip = Math.hypot(thumbTip.x - wrist.x, thumbTip.y - wrist.y);
  const dIp = Math.hypot(thumbIp.x - wrist.x, thumbIp.y - wrist.y);
  const dMcp = Math.hypot(thumbMcp.x - wrist.x, thumbMcp.y - wrist.y);
  
  // Thumb extended if tip is significantly farther than IP/MCP
  if (dTip > dIp * 1.15 && dTip > dMcp * 1.2) {
    count++;
  }
  
  return Math.min(5, Math.max(0, count));
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
    challengeType: null, // 'blink' | 'fingers'
    // For finger stability check
    fingerCountHistory: [],
    fingerStableSince: null,
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
        modelComplexity: 0,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });
      hands.onResults((results) => {
        const l = livenessRef.current;
        if (!l.active || l.challengeType !== "fingers") return;
        if (results.multiHandLandmarks?.length > 0) {
          const count = countExtendedFingers(results.multiHandLandmarks[0]);
          setFingersShownRef.current?.(count);
          
          // Stability check: require count to match required for 500ms
          const now = Date.now();
          if (count === l.required) {
            if (!l.fingerStableSince) {
              l.fingerStableSince = now;
            } else if (now - l.fingerStableSince >= 500) {
              // Stable for 500ms - complete!
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

          // Liveness: blink detection when challenge is active
          const l = livenessRef.current;
          if (l.active && l.challengeType === "blink" && l.required > 0) {
            try {
              const ear = getEAR(landmarks);
              const now = Date.now();
              
              // Detect eye closure
              if (ear < EAR_CLOSED && !l.eyesClosed) {
                l.eyesClosed = true;
              }
              // Detect eye opening after closure (blink complete)
              else if (ear >= EAR_OPEN && l.eyesClosed && now - l.lastBlinkTime > BLINK_COOLDOWN_MS) {
                l.eyesClosed = false;
                l.lastBlinkTime = now;
                l.count += 1;
                setLivenessCountRef.current?.(l.count);
                
                if (l.count >= l.required) {
                  l.active = false;
                  l.challengeType = null;
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
    // More varied randomization: 40% blink, 60% fingers (fingers are more reliable)
    // Also vary blink count (1-3) and finger count (1-5)
    const rand = Math.random();
    
    if (rand < 0.4) {
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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white flex flex-col items-center justify-center p-4 sm:p-6">
      {/* Liveness check overlay */}
      {livenessActive && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-gray-800 border border-gray-600 rounded-2xl p-8 max-w-md w-full shadow-2xl text-center animate-fade-in">
            <h3 className="text-xl font-bold text-white mb-2">Liveness check</h3>
            <p className="text-gray-300 mb-1">We need to confirm you’re not a photo or screen.</p>
            {livenessChallengeType === "fingers" && (
              <div className="text-6xl font-bold text-amber-400 my-4">{livenessRequired}</div>
            )}
            <p className="text-amber-400 font-semibold mb-6">{livenessPrompt}</p>
            {livenessChallengeType === "blink" && (
              <div className="mb-6">
                <div className="text-4xl font-bold text-blue-400 mb-1">
                  {livenessCount} / {livenessRequired}
                </div>
                <div className="text-sm text-gray-400">Blinks detected</div>
                {livenessCount < livenessRequired && (
                  <div className="mt-2 text-xs text-gray-500">
                    👁️ Blink naturally - close and open your eyes
                  </div>
                )}
              </div>
            )}
            {livenessChallengeType === "fingers" && (
              <div className="mb-6">
                <div className="text-4xl font-bold text-amber-400 mb-1">
                  {fingersShown} / {livenessRequired}
                </div>
                <div className="text-sm text-gray-400">Fingers shown</div>
                {fingersShown === livenessRequired && (
                  <div className="mt-2 text-xs text-green-400 animate-pulse">
                    ✓ Hold steady for a moment...
                  </div>
                )}
                {fingersShown !== livenessRequired && fingersShown > 0 && (
                  <div className="mt-2 text-xs text-yellow-400">
                    Adjust to show {livenessRequired} finger{livenessRequired === 1 ? "" : "s"}
                  </div>
                )}
              </div>
            )}
            <p className="text-sm text-gray-500 mb-6">
              Look at the camera and perform the action above.
            </p>
            <button
              type="button"
              onClick={cancelLiveness}
              className="px-4 py-2 rounded-lg border border-gray-500 text-gray-300 hover:bg-gray-700"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Toast Notification */}
      {toast.show && (
        <div
          className={`fixed top-4 right-4 z-50 px-6 py-3 rounded-lg shadow-xl backdrop-blur-md border transition-all duration-300 ${
            toast.type === "success"
              ? "bg-green-500/90 border-green-400 text-white"
              : "bg-red-500/90 border-red-400 text-white"
          } animate-slide-in`}
        >
          <div className="flex items-center gap-2">
            <span>{toast.type === "success" ? "✓" : "✗"}</span>
            <span className="font-medium">{toast.message}</span>
          </div>
        </div>
      )}

      <div className="bg-gray-800/70 backdrop-blur-xl shadow-2xl rounded-3xl p-6 sm:p-10 w-full max-w-2xl border border-gray-700/50 animate-fade-in">
        <div className="text-center mb-6">
          <h1 className="text-4xl sm:text-5xl font-bold mb-2 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            LUMA-X
          </h1>
          <p className="text-gray-400 text-sm sm:text-base">
            Multi-Modal Human Presence Attestation
          </p>
        </div>

        {/* Camera Section */}
        <div className="mb-6 flex flex-col items-center">
          <div className="relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="rounded-xl border-2 border-gray-600 w-full max-w-xs shadow-lg"
            />
            {cameraOn && (
              <div className="absolute top-2 right-2 flex items-center gap-2">
                <div
                  className={`w-3 h-3 rounded-full ${
                    faceDetected ? "bg-green-500 animate-pulse" : "bg-red-500"
                  }`}
                />
                <span className="text-xs bg-black/50 px-2 py-1 rounded">
                  {faceDetected ? "Face Detected" : "No Face"}
                </span>
              </div>
            )}
          </div>

          <div className="flex gap-3 mt-4">
            {!cameraOn ? (
              <button
                onClick={startCamera}
                className="bg-green-600 hover:bg-green-500 px-5 py-2.5 rounded-lg font-medium transition-all duration-200 hover:scale-105 active:scale-95 shadow-lg"
              >
                📷 Start Camera
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="bg-red-600 hover:bg-red-500 px-5 py-2.5 rounded-lg font-medium transition-all duration-200 hover:scale-105 active:scale-95 shadow-lg"
              >
                ⏹️ Stop Camera
              </button>
            )}
          </div>
        </div>

        {/* Typing Stats Display */}
        {typingStats && (
          <div className="mb-4 p-3 bg-gray-900/50 rounded-lg border border-gray-700/50">
            <div className="grid grid-cols-3 gap-2 text-xs sm:text-sm">
              <div className="text-center">
                <div className="text-gray-400">Keystrokes</div>
                <div className="font-bold text-blue-400">{keystrokes.length}</div>
              </div>
              <div className="text-center">
                <div className="text-gray-400">Mean Interval</div>
                <div className="font-bold text-purple-400">{typingStats.mean}s</div>
              </div>
              <div className="text-center">
                <div className="text-gray-400">Std Dev</div>
                <div className="font-bold text-pink-400">{typingStats.std}s</div>
              </div>
            </div>
          </div>
        )}

        <textarea
          className="w-full p-4 rounded-xl bg-gray-900/80 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 resize-none"
          rows="4"
          placeholder="Type here to capture keystroke patterns..."
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />

        {keystrokes.length === 0 && (
          <p className="mt-2 text-xs text-gray-500 text-center">
            Start typing to capture keystroke data
          </p>
        )}

        <div className="mt-6 space-y-3">
          <button
            onClick={handleVerify}
            disabled={loading.verify || !faceDetected || keystrokes.length < 2}
            className="w-full bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed p-3.5 rounded-xl font-semibold transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] shadow-lg flex items-center justify-center gap-2"
          >
            {loading.verify ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Verifying...
              </>
            ) : (
              <>
                🔍 Verify Human Presence
                <span className="text-xs opacity-70 ml-auto">Ctrl+Enter</span>
              </>
            )}
          </button>

          <button
            onClick={handleCollect}
            disabled={loading.collect || !faceDetected || keystrokes.length < 2}
            className="w-full bg-gradient-to-r from-purple-600 to-purple-500 hover:from-purple-500 hover:to-purple-400 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed p-3.5 rounded-xl font-semibold transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] shadow-lg flex items-center justify-center gap-2"
          >
            {loading.collect ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Collecting...
              </>
            ) : (
              <>
                💾 Collect Human Training Sample
                <span className="text-xs opacity-70 ml-auto">Ctrl+S</span>
              </>
            )}
          </button>

          <button
            onClick={handleDemoBot}
            disabled={loading.demo}
            className="w-full bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed p-3.5 rounded-xl font-semibold transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] shadow-lg flex items-center justify-center gap-2"
          >
            {loading.demo ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Loading...
              </>
            ) : (
              <>
                🤖 Demo: Show Bot Result
              </>
            )}
          </button>

          <button
            type="button"
            onClick={resetForm}
            className="w-full mt-2 p-2.5 rounded-xl font-medium border border-gray-600 text-gray-300 hover:bg-gray-700/50 hover:border-gray-500 transition-all duration-200 flex items-center justify-center gap-2"
          >
            🔄 Refresh (clear form & result)
          </button>
        </div>

        {result && (
          <div className="mt-6 bg-gradient-to-br from-gray-900 to-gray-800 p-5 rounded-xl border border-gray-700/50 animate-fade-in shadow-xl">
            {/* Probability Bar */}
            {typeof result.human_probability === "number" && (
              <div className="mb-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-gray-300">Human Probability</span>
                  <span className="text-lg font-bold text-blue-400">
                    {(result.human_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 rounded-full ${
                      result.prediction === 1
                        ? "bg-gradient-to-r from-green-500 to-green-400"
                        : "bg-gradient-to-r from-red-500 to-red-400"
                    }`}
                    style={{ width: `${result.human_probability * 100}%` }}
                  />
                </div>
              </div>
            )}

            {/* Prediction */}
            <div className="text-center mb-4">
              <div className="text-xs text-gray-400 mb-1">PREDICTION</div>
              <div className={`text-2xl font-bold ${
                result.prediction === 1 ? "text-green-400" : "text-red-400"
              }`}>
                {result.prediction === 1 ? (
                  <span className="flex items-center justify-center gap-2">
                    <span>✅ Human Verified</span>
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    <span>❌ Bot Detected</span>
                  </span>
                )}
              </div>
            </div>

            {/* Hash */}
            {result.hash && (
              <div className="pt-4 border-t border-gray-700">
                <div className="text-xs text-gray-400 mb-1">VERIFICATION HASH</div>
                <div className="text-xs font-mono break-all text-gray-300 bg-gray-900/50 p-2 rounded">
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
