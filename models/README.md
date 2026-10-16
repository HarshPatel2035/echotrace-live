# Local hand model

`hand_landmarker.task` is the MediaPipe Hand Landmarker float16 model bundle, version 1. It contains the palm detector and hand-landmark estimator and is loaded from disk only when G enables gestures. Inference runs locally on CPU. EchoTrace has no runtime download or networking code.

Source: [Google's official versioned model bundle](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).

Documentation: [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).

Retain this file and the model when copying the project. If you deliberately remove it, keyboard effects still run; G reports that the local model is missing.
