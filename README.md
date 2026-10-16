# EchoTrace Live

Paint neon light in the air with your webcam. Moving hands, faces, and objects leave glowing echoes that linger and slowly fade. Five live modes combine motion detection, colorful trails, soft bloom, futuristic edges, and a mild vignette.

Everything runs on your laptop using Python, OpenCV, and NumPy. Optional MediaPipe Hands adds gestures and sounddevice adds microphone-reactive effects. No cloud, servers, accounts, API keys, or GPU required. Installation needs internet to download dependencies; running afterward does not. The hand model is included in `models/hand_landmarker.task`. No video is saved unless you press **S** or **V**. Recordings have no audio; the microphone is used only for live analysis, and only after pressing A. Gestures and audio start OFF.

## Setup and run

Use Python **3.10 or newer** and a desktop session with a working webcam. Open a terminal in the project folder containing `main.py`.

For an existing installation, activate your environment and run:

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
pip install --upgrade -r requirements.txt
pip install --force-reinstall --no-deps "opencv-contrib-python>=4.8,<5"
python main.py
```

Only one OpenCV distribution should own `cv2`. This project now uses the desktop `opencv-contrib-python` wheel because MediaPipe depends on it; it includes all the regular OpenCV functions. The forced reinstall repairs files shared by the old wheel during migration. For a clean environment, follow the normal commands below.

### Windows (Command Prompt)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

In PowerShell, activate with `venv\Scripts\Activate.ps1`. If script execution is blocked, use Command Prompt with the commands above, or run `venv\Scripts\python.exe main.py` after installing with `venv\Scripts\python.exe -m pip install -r requirements.txt`.

### macOS

On systems where Python is named `python3`, substitute `python3` for the first command. After activation, `python` points to the virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Linux

Use the same commands (substitute `python3` for the first command if needed):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

On Debian/Ubuntu, missing virtual-environment or desktop libraries can be installed with `sudo apt install python3-venv libgl1 libglib2.0-0`. Run from a graphical desktop, not a headless SSH session.

Focus the **EchoTrace Live** window and move to create trails. Quit before closing your terminal. Activate the environment again when returning to the project; installation is only needed once.

### Command-line options

All options are optional; `python main.py --help` lists them.

| Option | Default | Purpose |
| --- | --- | --- |
| `--camera N` | 0 | Webcam index; try 1, 2 … on computers with several cameras |
| `--width W --height H` | 1280×720 | Requested capture size; `0` keeps the camera's own default |
| `--fps N` | 30 | Requested camera frame rate; `0` keeps the camera default |
| `--work-width N` | 640 | Maximum width for effect processing; 480 is faster on slow laptops |
| `--preset 1-5` | none | Start with one of the visual presets below |
| `--no-mirror` | mirrored | Start without the horizontal mirror |
| `--hud-in-captures` | off | Include the HUD in snapshots and recordings (also toggled with I) |
| `--flicker-limit F` | 0.45 | Ignore frames where more than this fraction of the picture changes at once; `0` disables |
| `--output-dir DIR` | `captures/` | Where snapshots and recordings are saved |

Example for a slower laptop with a second camera: `python main.py --camera 1 --width 640 --height 480 --work-width 480 --preset 2`

## Controls

| Key | Action |
| --- | --- |
| Q / Esc | Quit and finalize any recording |
| S | Save a PNG snapshot |
| V | Start / stop recording |
| R | Clear the persistent trails |
| C | Cycle Echo → Neon → Ghost → Fire → Rainbow |
| + / - | Increase / decrease trail retention (decay value) |
| [ / ] | Decrease / increase glow |
| M | Toggle horizontal mirror and clear trails |
| Space | Pause / resume effects |
| H | Show / hide key help (hidden at start) |
| G | Enable / disable MediaPipe hand gestures |
| A | Enable / disable default microphone audio response |
| P | Enable / disable motion particles |
| K | Cycle symmetry: off → 2 → 4 → 6 → 8 → off |
| T | Enable / disable vertical time-slice / slit-scan |
| F | Toggle frozen effect image |
| 1–5 | Apply a visual preset and clear the canvas |
| Z | Hide / show the entire HUD |
| O | Hands-only painting: only tracked hands leave trails (starts gestures if needed) |
| I | Include / exclude the HUD in snapshots and recordings (excluded by default) |

The `=` key also increases retention, so you do not need Shift on a standard keyboard. Letter controls accept upper and lower case. The HUD starts compact (status only); press H for the full key list. Default decay is **0.960** and glow is **1.4**. Higher decay means **slower fading / longer trails**; it ranges from 0.800 to 0.995. Glow ranges from 0.0 to 4.0. Pausing freezes the picture and trails while keeping controls active and the camera connection fresh. During a pause, recording continues with the frozen image. Reset takes effect on the next processed frame after resume. Switching modes preserves existing trails for mixed-color transitions; press R for a clean start. Space pause, F freeze, and fist freeze are independent: release all active holds to resume. Hand detection continues during every freeze so releasing a fist resumes automatically.

## Hand gestures

Press G and show your hand to the camera. MediaPipe's CPU Hand Landmarker uses the included local model; no network code runs in the application. The Tasks API is used to support current MediaPipe versions, without relying on the older `mp.solutions` API.

- **Pinch thumb and index:** advance the color mode once. Release the pinch before repeating.
- **Open palm (four fingers extended):** clear trails, particles, rings, and slit-scan history once. Release before repeating.
- **Fist (four fingers curled):** hold to freeze; open your hand or remove it to resume. Tracking loss also releases the gesture freeze.
- **Two visible hands:** temporarily enable six-way kaleidoscope if keyboard symmetry is off; otherwise keep the symmetry selected with K. Remove one hand to restore the keyboard setting.
- **Index finger extended, other fingers curled:** draw a neon line from the fingertip. Curl the index or remove the hand to lift the pen. Each hand has its own pen path.

**Hands-only painting (O):** normally anything that moves paints, so small head or body movements light up your whole outline. Press O to paint only inside the tracked hand areas (a padded outline around each detected hand). It turns gestures on automatically if they are off, and turns off again when you press G to stop gestures. When no hand has been seen for 0.6 seconds nothing paints, so trails fade out cleanly. The gestures above stay active in this mode, so a fist still freezes.

Pinch takes priority over palm/fist recognition. Mode/reset actions require two consecutive classifications plus an 0.8-second cooldown to avoid repeats. Gestures are geometric heuristics, so face the palm toward the webcam, use good lighting, and avoid overlapping hands. Inference uses a single isolated local process at up to 12 Hz and 320-pixel input width; no inference queue can build up. Native tracker failures disable gestures without closing the main app. Toggle G off to release the tracker. First enablement can take a few seconds while dependencies/model initialize. The file `models/README.md` records the model source.

## Audio response

Press A to open the **default input microphone**. This reacts to nearby music or your voice, not direct computer playback unless your operating system already provides a local loopback input. Use headphones or moderate speaker volume to avoid feedback in other audio apps; EchoTrace itself never plays audio.

A 2048-sample FFT measures bass (40–250 Hz) and treble (2–10 kHz). Bass boosts glow above the keyboard setting, capped at 4.0. Treble adds brighter outlines and edge sparkles. A bass-energy rise above an adaptive baseline creates expanding shockwave rings with a 0.3-second refractory period. Ring lifetimes/counts are capped, silence fades the response, and audio samples are held only in a short RAM queue. No audio is recorded or sent anywhere. Audio remains enabled when switching visual presets. During freeze, input analysis continues but no new rings are emitted.

## Particles, kaleidoscope, and time-slice

P spawns particles on moving contours. Velocity, gravity, lifetime, and fading run as NumPy array operations, with a maximum of 450 particles and 100 spawns/second. Glowing particles and audio rings blend additively into the decaying trail canvas; only bounded contour/ring drawing uses small loops.

K applies a cached radial mirror map with 2, 4, 6, or 8 sectors to the persistent trail canvas, including finger lines, particles, and shockwaves. The live camera base remains visible behind the symmetric trails. This does not repeatedly transform the stored canvas, so cycling symmetry preserves the original motion history.

T builds a vertical slit-scan: a narrow slice travels left-to-right, replacing those columns with the latest composite while other columns retain earlier moments. A sweep takes approximately four seconds at 30 FPS and wraps around. Motion becomes a stretched time mosaic. R resets its history; T off restores the live effect immediately.

| Preset | Color mode | Particles | Symmetry | Slit-scan | Decay / glow |
| --- | --- | --- | --- | --- | --- |
| 1 — Classic Echo | Echo | Off | Off | Off | 0.960 / 1.4 |
| 2 — Neon Sparks | Neon | On | Off | Off | 0.975 / 1.8 |
| 3 — Rainbow Mandala | Rainbow | On | 6 | Off | 0.970 / 1.6 |
| 4 — Ghost Time | Ghost | Off | Off | On | 0.985 / 1.0 |
| 5 — Fire Vortex | Fire | On | 4 | Off | 0.950 / 2.0 |

Presets leave microphone/gesture enablement, mirror, HUD visibility, and freeze controls unchanged. Device access is always controlled by A and G.

## The five effects

- **Echo:** classic persistent trails with a slowly cycling neon color.
- **Neon:** bold cyan/magenta motion with stronger luminous outlines.
- **Ghost:** softly glowing, translucent echoes of the moving camera image.
- **Fire:** hot red, orange, and yellow traces.
- **Rainbow:** continuously shifting spatial rainbow hues that linger in the canvas.

For light-painting, try a darker background with a well-lit hand or a small flashlight. Keep the camera steady: moving the camera changes the entire image and paints large trails. Exposure changes can also generate traces because motion is measured with frame differences.

## Captures

The app creates `captures/` **beside `main.py`** when you first save or record. Snapshots and recordings contain only the effect image without the HUD; press I (or start with `--hud-in-captures`) to include it. Videos use MP4 with the `mp4v` codec first, then automatically try XVID/AVI and MJPG/AVI if a writer cannot open. Filenames include timestamps to avoid overwriting captures. Press V again or quit normally to finalize the file. Output codecs depend on your local OpenCV build; recordings are paced at a fixed rate based on measured FPS at start (5–30 FPS). Very long camera or disk stalls can shorten playback relative to elapsed wall time. OpenCV cannot reliably report every later disk/encoder failure, so check your first short recording before a long session. Odd frame dimensions are padded by one pixel for codec compatibility.

## Performance and implementation

The camera receives requests for **1280×720 at 30 FPS**. If unsupported, the app uses whatever dimensions the camera actually returns without stretching to 720p. Preview is mirrored by default. Motion detection, the persistent floating-point canvas, glow, and edges run at up to 640 pixels wide (`--work-width`). Only that light overlay is enlarged; it is blended onto the full-resolution camera image so your face and the room stay sharp. Camera reading, effects, and window display each take roughly 15 ms on a laptop, so they run on separate threads: a camera thread keeps only the newest frame, an effects thread processes it (and applies key presses between frames), and the main thread only shows finished frames, which macOS requires. This keeps up with a 30 FPS camera; the HUD shows the effects rate, and a frame reaches the screen about one frame later. All pixel work uses vectorized NumPy/OpenCV operations. The display loop targets 30 FPS; hardware, webcam exposure, and recording speed affect actual FPS.

Each frame is blurred and differenced against the previous frame. A threshold and small morphological filter suppress noise. If more than 45% of the picture changes at once (auto-exposure, lights switching, a bumped camera) that frame's motion is ignored instead of flooding the canvas, and the HUD says so. In Neon mode the unthresholded difference sets each pixel's colour, so slower or fainter motion glows cyan and faster motion glows magenta. Motion is colorized and blurred into a glow, then merged into a time-decaying canvas with lighten/max blending. Subtle Canny edges and a vignette complete the look. Decay is adjusted for elapsed time so trail duration stays consistent at different frame rates.

`main.py` parses the command-line options and starts the app. The code lives in the `echotrace/` package: `app.py` (the `EchoTraceAgent` class: camera, frame pipeline, keys, main loop), `effects.py` (motion, colour modes, glow, particles, symmetry, slit-scan), `camera.py` (background frame reader), `gestures.py` (hand tracker worker, gestures, hands-only mask), `audio.py` (microphone analysis), `capture.py` (snapshots and recording), `hud.py` (on-screen panel), and `paths.py` (project folders). The small hand model is loaded only after G. MediaPipe and sounddevice imports are lazy: if installation/device access fails, the affected feature reports a message and the core webcam app continues.

## Troubleshooting

### Webcam permissions / camera cannot open

- **Windows:** Settings → Privacy & security → Camera: enable camera access and access for desktop apps. Close Camera, Teams, Zoom, browser camera tabs, and other capture programs. The app tries `cv2.VideoCapture(0)` first and DirectShow on Windows if opening fails.
- **macOS:** System Settings → Privacy & Security → Camera: allow the terminal or app launching Python (Terminal, iTerm, your IDE, or Codex). Restart that app after changing permissions. Accept the camera permission prompt if shown.
- **Linux:** check `/dev/video0`, desktop camera permissions, and user access to the camera device. Close competing capture apps. The default camera index is 0; on a computer with several cameras, start with `python main.py --camera 1` (or 2, 3 …).

### Black screen or no frames

- Remove the camera cover, check lighting, and test the webcam in your operating system's camera app, then close that app.
- Check privacy permissions and reconnect external webcams. Thirty consecutive failed reads stop the app with a clear error.
- Some cameras reject resolution requests. Start with `python main.py --width 0 --height 0 --fps 0` to use the camera's default capture configuration.
- A dark background without movement produces little glow by design. Move a well-lit hand, try Rainbow, increase glow with `]`, and clear stale trails with R.
- If the window fails to appear, use the desktop `opencv-contrib-python` wheel, not a headless wheel. In the activated environment, follow the existing-installation repair commands above. Never install both OpenCV wheel families together.

### Lag / low FPS

- Add lighting: long auto-exposure can reduce camera FPS. Close other camera apps and CPU-intensive programs.
- Lower glow with `[`, stop recording with V, and try Echo instead of Neon.
- Disable gestures (G) and audio (A) if they compete for CPU. Turn off particles (P), symmetry (K until OFF), or slit-scan (T) individually. Frame processing targets 30 FPS, but enabling every feature and recording simultaneously depends on your laptop.
- For slower hardware, run `python main.py --width 640 --height 480 --work-width 480`.
- In dim rooms many webcams lower their own frame rate (often to 15 FPS) to get longer exposures. More light usually raises the FPS shown in the HUD.
- A reported 30 FPS is a target, not a guarantee. USB bandwidth, camera drivers, desktop display, and disk speed matter.

### Snapshots or recording fail

Ensure the project directory is writable and the disk has free space. The console and HUD report failures and the selected fallback codec. If all codecs fail to open, install the standard desktop OpenCV wheel in a clean virtual environment. Stop recording normally before moving files or powering down.

### Controls do nothing / closing the window

Click inside the OpenCV window to give it keyboard focus. Use Q or Esc for reliable shutdown; closing the window also exits when the GUI backend reports it closed. Ctrl+C is supported in the terminal. The camera, writer, and windows are released in cleanup.

### Hand tracking unavailable

Keep `models/hand_landmarker.task` with the project. Reinstall `mediapipe>=0.10.31,<0.11` if it is missing or incompatible. Some platforms/Python releases do not have a compatible MediaPipe wheel; use a supported 64-bit Python environment (3.11 or 3.12 are practical fallback choices). For the core app without optional features, install `numpy` and `opencv-contrib-python` only. G reports the failure while keyboard controls and other visual effects keep working. No model is automatically downloaded at runtime.

On macOS, MediaPipe's native package can initialize graphics services even with its inference delegate set to CPU. Launch from a normal graphical desktop terminal; a restricted/headless environment may terminate the tracker process. If this happens, the app disables gestures and reports it. The other effects remain available.

### Microphone unavailable / no audio response

- **macOS:** System Settings → Privacy & Security → Microphone: enable access for the application launching Python and restart it.
- **Windows:** Settings → Privacy & security → Microphone: enable access for desktop apps and select the intended default input device.
- **Linux:** select a working input in your desktop audio settings. If PortAudio is missing on Debian/Ubuntu, run `sudo apt install libportaudio2`; other distributions provide an equivalent PortAudio runtime package.
- Close programs exclusively using the microphone. List local devices with `python -m sounddevice`. Ensure the default input is receiving sound; audio is not system loopback by default. Very quiet inputs may not pass the beat threshold, so try a clear low-frequency pulse or increase the system input gain.
- If sounddevice cannot import, run `pip install "sounddevice>=0.5,<0.6"`. Errors are shown without stopping the webcam. Toggle A off to release the microphone.

Implementation references: [MediaPipe Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) and [sounddevice InputStream API](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html).

## Hardware-free checks

In the activated environment:

```bash
python -m unittest -v test_effects.py
```

These tests cover motion/decay across five modes, gesture classification and debounce, freeze/resume, bass/treble analysis and beats, particle lifecycles, kaleidoscope symmetry, slit-scan history, presets/HUD, optional dependency and worker failures, PNG saving, video decode, Neon speed colours, the whole-frame flicker guard, full-resolution output, hands-only painting, HUD-free captures, and command-line options. They do not access the webcam or microphone. Actual hand-pose accuracy and end-to-end FPS depend on local devices and lighting.
