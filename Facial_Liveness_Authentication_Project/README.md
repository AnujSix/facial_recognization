# Real-Time Facial Liveness Authentication

A TY CSE project prototype that performs local face enrollment, randomized active
liveness challenges, face matching, and authentication logging.

## What it demonstrates

1. Detect one face from a webcam with MediaPipe Face Mesh.
2. Ask for two random actions: blink, turn left, or turn right.
3. Match the live face against a locally enrolled LBPH face profile.
4. Grant access only when both liveness and identity checks pass.
5. Record decisions in `data/authentication_log.csv`.

No uploaded JPG is required. This avoids failures caused by missing paths such as
`upload/08-1000164403.jpg` and `upload/09-1000164403.jpg`.

## Setup

Use Python 3.10 or 3.11. From this folder:

### Windows

Using the quick launcher:
```powershell
.\run.bat demo                          # Run automated end-to-end verification
.\run.bat enroll --name "Your Name"    # Enroll your face using webcam
.\run.bat authenticate                  # Test active liveness & authentication
```

Or manually with python:
```powershell
.venv\Scripts\activate
python app.py enroll --name "Shrikant Sarode"
python app.py authenticate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

```

Press `Q` or Escape to close the camera. If camera 0 is wrong, place `--camera 1`
before the command, for example `python app.py --camera 1 authenticate`.

## Demo plan

- Enroll the authorized person under normal lighting.
- Run authentication with the real person; complete the random prompts.
- Try a printed photograph. It cannot blink or follow a random prompt, so it times out.
- Try a still face on another phone. It should fail for the same reason.
- Show `data/authentication_log.csv` as the experiment record.

The default LBPH distance threshold is 70; lower values are stricter. Test several
authorized and unauthorized people and choose a threshold based on your own camera.

## Important research limitation

This is an educational active-liveness prototype, not a production security system.
A prepared replay video or deepfake may imitate movements. Stronger resistance needs
a trained passive anti-spoofing model (for example MiniFASNet/MultiFTNet), depth or IR
hardware where available, and evaluation on datasets such as OULU-NPU, Replay-Attack,
CASIA-FASD, or CelebA-Spoof. Do not use this prototype to protect real accounts or
sensitive facilities.
only run this:
python app.py enroll --name "Shrikant Sarode"
python app.py authenticate
