import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { FaceMesh } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";

function App() {
  const [content, setContent] = useState("");
  const [keystrokes, setKeystrokes] = useState([]);
  const [cameraOn, setCameraOn] = useState(false);
  const [result, setResult] = useState(null);
  const [faceVector, setFaceVector] = useState([]);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const faceMeshRef = useRef(null);

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
        }
      });

      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          await faceMesh.send({ image: videoRef.current });
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
      videoRef.current.srcObject = null;
      setCameraOn(false);
    }
  };

  // ---------- Keystroke Capture ----------
  useEffect(() => {
    const handleKey = () => {
      setKeystrokes((prev) => [...prev, Date.now()]);
    };

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
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
      alert("Face not detected properly.");
      return null;
    }

    return [...faceVector, ...typingVector]; // 103 features
  };

  // ---------- Verify ----------
  const handleVerify = async () => {
    const sessionVector = buildSessionVector();
    if (!sessionVector) return;

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/verify",
        {
          session_vector: sessionVector,
          content: content,
        }
      );

      setResult(response.data);
    } catch (error) {
      console.error("Backend error:", error);
    }
  };

  // ---------- Collect Human Training Sample ----------
  const handleCollect = async () => {
    const sessionVector = buildSessionVector();
    if (!sessionVector) return;

    try {
      await axios.post(
        "http://127.0.0.1:8000/collect",
        {
          session_vector: sessionVector,
          content: "training",
          label: 1,  // ✅ Human label
        }
      );

      alert("Human session stored successfully!");
    } catch (error) {
      console.error("Collect error:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white flex flex-col items-center justify-center p-6">
      <div className="bg-gray-800/60 backdrop-blur-md shadow-2xl rounded-2xl p-10 w-full max-w-xl border border-gray-700">
        <h1 className="text-4xl font-bold text-center mb-2">LUMA-X</h1>

        <p className="text-gray-400 text-center mb-6">
          Multi-Modal Human Presence Attestation
        </p>

        <div className="mb-4 flex justify-center">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="rounded-lg border border-gray-600 w-64"
          />
        </div>

        <div className="flex gap-4 justify-center mb-6">
          {!cameraOn ? (
            <button
              onClick={startCamera}
              className="bg-green-600 hover:bg-green-500 px-4 py-2 rounded"
            >
              Start Camera
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="bg-red-600 hover:bg-red-500 px-4 py-2 rounded"
            >
              Stop Camera
            </button>
          )}
        </div>

        <textarea
          className="w-full p-4 rounded-lg bg-gray-900 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows="4"
          placeholder="Enter content to seal..."
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />

        <p className="mt-2 text-sm text-gray-400">
          Keystrokes captured: {keystrokes.length}
        </p>

        <button
          onClick={handleVerify}
          className="mt-6 w-full bg-blue-600 hover:bg-blue-500 p-3 rounded-lg font-semibold"
        >
          Verify Human Presence
        </button>

        <button
          onClick={handleCollect}
          className="mt-3 w-full bg-purple-600 hover:bg-purple-500 p-3 rounded-lg font-semibold"
        >
          Collect Human Training Sample
        </button>

        {result && (
          <div className="mt-6 bg-gray-900 p-4 rounded-lg text-sm break-all">
            <p>
              <strong>Human Probability:</strong>{" "}
              {result.human_probability.toFixed(3)}
            </p>

            <p className="mt-3 text-lg font-bold">
              <strong>Prediction:</strong>{" "}
              {result.prediction === 1 ? (
                <span className="text-green-400">Human ✅</span>
              ) : (
                <span className="text-red-400">Bot ❌</span>
              )}
            </p>

            <p className="mt-4">
              <strong>Hash:</strong> {result.hash}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
