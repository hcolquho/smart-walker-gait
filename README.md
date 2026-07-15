# Smart Walker Gait Analysis Pipeline

Modular RGB-D gait analysis system for the Evolution Mini Trillium rollator.
Pipeline: `FrameSource → Detector → Tracker → PoseEstimator2D → Depth → Smoother → GaitEvents → GaitMetrics`

Every stage is an abstract interface with swappable backends (registered via
a small factory pattern in `core/factory.py`), selected by name in
`configs/pipeline.yaml`. The whole pipeline runs today against synthetic /
mock data with **zero hardware and zero heavy ML dependencies** — the Femto
Bolt, RTMPose/ViTPose, and MotionBERT backends are stubbed with clear wiring
instructions and drop in later without touching any other stage.

---

## Repo structure

```
smart_walker_gait/
├── README.md
├── requirements.txt
├── pyproject.toml                    # pytest config, src-layout packaging
├── configs/
│   ├── pipeline.yaml                 # master config: backend selection per stage
│   └── camera_intrinsics_default.yaml
├── src/walker_gait/
│   ├── core/
│   │   ├── types.py                  # Frame, Detection, Skeleton2D/3D, GaitEvent, GaitMetrics, ...
│   │   └── factory.py                # Registry: generic build-from-YAML pattern
│   ├── frame_source/
│   │   ├── base.py                   # FrameSource ABC
│   │   ├── synthetic.py              # SyntheticSource -- your test harness, build first
│   │   ├── webcam.py
│   │   ├── video_file.py
│   │   └── mono_depth_proxy.py       # monocular-depth stand-in until the Bolt arrives
│   ├── tracking/
│   │   ├── base.py
│   │   └── iou_tracker.py            # IOU-based, adds track_id
│   ├── detector/
│   │   ├── base.py
│   │   ├── dummy.py                  # `dummy` (fixed bbox) + `motion` (bg-subtraction) -- zero-dependency
│   │   └── yolov8_backend.py         # real YOLOv8 -- stub, needs `pip install ultralytics`
│   ├── pose2d/
│   │   ├── base.py
│   │   ├── dummy.py                  # zero-dependency mock, works today
│   │   └── mmpose_backend.py         # RTMPose (live) / ViTPose (offline) -- stub, needs mmpose
│   ├── depth/
│   │   └── backprojection.py         # median-window pinhole backprojection, fully real & tested
│   ├── smoothing/
│   │   ├── base.py
│   │   ├── kalman.py                 # live tier: causal, zero added latency
│   │   └── motionbert_backend.py     # offline tier -- stub, needs MotionBERT repo + checkpoint
│   ├── gait/
│   │   ├── events.py                 # heel-strike / toe-off detection (+ optional FSR corroboration)
│   │   └── metrics.py                # cadence, asymmetries, double support, aggregate score
│   ├── imu/
│   │   ├── base.py
│   │   ├── dummy.py
│   │   └── bno085_backend.py         # stub -- needs Adafruit BNO08x library + wiring
│   └── calibration/
│       └── calibrate_camera.py       # OpenCV checkerboard calibration script
├── scripts/
│   └── run_pipeline_demo.py          # runs the FULL pipeline today, synthetic end-to-end
└── tests/
    ├── test_synthetic_source.py
    ├── test_tracker.py
    ├── test_backprojection.py
    ├── test_kalman_smoother.py
    ├── test_gait_events.py
    └── test_gait_metrics.py
```

### Detector module
Implemented in this repo now (`detector/base.py`, `detector/dummy.py`,
`detector/yolov8_backend.py`) — `dummy` and `motion` work with zero extra
dependencies today, `yolov8` needs `pip install ultralytics` (see
requirements.txt for the note on why that's a heavier, separate install). If
you already had a different YOLOv8 wrapper from earlier work, just swap its
internals into `Yolov8Detector.detect()` — the `Detector` ABC / registry
pattern and the `Detection` output type are what everything downstream
depends on, not the specific class name.

---

## Setup

```bash
cd smart_walker_gait
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
pip install -e .                                       # makes `walker_gait` importable
```

## Run the full pipeline today (no hardware, no heavy ML deps)

```bash
python scripts/run_pipeline_demo.py
```

This runs `SyntheticSource → IouTracker → (mock 2D pose) → DepthBackprojector
→ KalmanSkeletonSmoother → VerticalVelocityEventDetector → GaitMetricsCalculator`
end-to-end and prints computed gait metrics next to the ground-truth
asymmetry that was deliberately injected, so you can sanity-check the whole
chain at a glance.

## Try the detector + tracker on your real webcam (no weights needed)

```bash
python scripts/run_pipeline_webcam.py --detector motion
```

Opens a window showing your webcam feed with the detector's bbox, the
tracker's `track_id`, and mock 2D keypoints overlaid — press `q` to quit.
`--detector motion` reacts to actual movement (background subtraction, zero
ML weights); `--detector dummy` always boxes a fixed centered region. This is
the fastest way to confirm the detector→tracker→pose2d wiring works on real
video before YOLOv8 weights are downloaded.

---

## Tests to run, and what each one is actually checking

Run everything with:
```bash
pytest tests/ -v
```

| Test file | What it validates | Why it matters |
|---|---|---|
| `test_synthetic_source.py` | The synthetic ground-truth 2D projection exactly matches the pinhole projection of the ground-truth 3D skeleton; the depth channel matches ground-truth z at joint pixels; injecting `left_right_asymmetry` actually perturbs the leg trajectories. | **This is the harness every other test leans on.** If this is wrong, every other "validated against synthetic ground truth" test is meaningless. |
| `test_detector.py` | `DummyDetector` returns a bbox that scales with frame size; `MotionDetector` reports nothing against a static background and correctly boxes an injected moving blob. | Confirms the detector stage's output contract (`Detection` bboxes) is sane before real YOLOv8 weights are in the loop — `MotionDetector` in particular is your first real-webcam sanity check. |
| `test_tracker.py` | `track_id` persists across small bbox shifts; survives a brief (few-frame) occlusion; is correctly dropped and reassigned once occlusion exceeds `max_misses`. | Confirms the tracker does its one job — bridging brief detector dropouts — without silently merging or losing identity, which would corrupt every downstream per-joint trajectory. |
| `test_backprojection.py` | Backprojection recovers ground-truth 3D position within tolerance on clean synthetic depth; a joint with zero valid depth is marked `invalid` rather than returning garbage coordinates; a single extreme noise-spike pixel doesn't corrupt the median-window estimate. | Depth accuracy is the piece most affected by real hardware — this is your regression baseline to re-run once the Femto Bolt's real depth is in the loop, to see how real noise compares to synthetic. |
| `test_kalman_smoother.py` | Smoothed error is lower than raw measurement error under injected Gaussian noise; the filter coasts through a short occlusion using its velocity estimate; it correctly marks a joint invalid once occlusion exceeds the configured coast budget. | This is the **live-path** smoother — confirms it actually reduces noise (not just relabels it) and handles occlusion sanely without silently freezing or snapping to zero. |
| `test_gait_events.py` | Detected heel-strike count roughly matches the expected count for a known injected cadence; FSR corroboration correctly snaps a vision-detected event onto a simulated FSR rising edge. | Validates the event detector against a *known* answer — something you cannot do once you're only working with real, unlabeled video. |
| `test_gait_metrics.py` | A perfectly symmetric hand-crafted event sequence yields near-zero asymmetry; a deliberately asymmetric one yields the correct sign and clearly nonzero magnitude; FSR-based loading asymmetry responds correctly to unequal per-side load; double support time matches a hand-computed expected value. | Confirms the metrics math itself is correct, independent of upstream vision/event-detection quality — the whole point of keeping `events.py` and `metrics.py` separate. |

All 19 tests pass as of this build (`pytest tests/ -v`).

**Before moving to real hardware**, the most valuable extra test to add
yourself: capture a short real webcam clip of you walking normally and with a
deliberate limp, and manually annotate a handful of heel-strike frames by eye
(scrub frame-by-frame). Compare against what `VerticalVelocityEventDetector`
reports on the same clip (once `pose2d` is switched from `dummy` to
`rtmpose`). This is your first real (non-synthetic) validation point, and
where you'll likely find the naive local-minima detector needs tuning or
replacing.

---

## Modules to install, in order

Only install a stage's heavy dependency when you're ready to move that stage
off its mock/synthetic backend — everything works without any of these until
then.

1. **Now, for local dev on any machine** (already in `requirements.txt`):
   ```bash
   pip install numpy opencv-python pyyaml pytest
   ```

2. **When you start developing against a live webcam** (already covered by
   `requirements.txt` via `opencv-python`) — no extra install needed,
   `WebcamSource`, `VideoFileSource`, `DummyDetector`, and `MotionDetector`
   all work out of the box. Try
   `python scripts/run_pipeline_webcam.py --detector motion` first.

3. **When you're ready to replace `dummy`/`motion` with real YOLOv8
   detection** (`detector/yolov8_backend.py`):
   ```bash
   pip install ultralytics
   ```
   First run auto-downloads `yolov8n.pt` if not already present locally —
   budget time/bandwidth, it pulls in torch as a dependency.

4. **When you want a simulated depth channel from RGB alone**
   (`MonoDepthProxySource`, useful for exercising the backprojection code
   path before the Bolt arrives):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install transformers timm pillow
   ```

5. **When you're ready to replace the dummy 2D pose estimator with RTMPose /
   ViTPose** (`pose2d/mmpose_backend.py`):
   ```bash
   pip install -U openmim
   mim install mmengine "mmcv>=2.0.1" mmpose mmdet
   ```
   Then pick and download a checkpoint (see docstring in
   `mmpose_backend.py` for suggested starting model names).

6. **When you're ready to wire up the Femto Bolt** (not implemented in this
   repo yet — you'll add `frame_source/femto_bolt.py`):
   ```bash
   # Orbbec SDK / pyorbbecsdk, per Orbbec's official install instructions
   # (also confirm Azure Kinect Body Tracking SDK if you use it for
   # comparison/cross-validation against your own pose pipeline)
   ```

7. **When you're ready to add offline MotionBERT smoothing**
   (`smoothing/motionbert_backend.py`):
   ```bash
   pip install torch torchvision
   git clone https://github.com/Walter0807/MotionBERT
   # download a pretrained checkpoint per their README
   ```

8. **When the BNO085 IMU hardware is wired up** (`imu/bno085_backend.py`):
   ```bash
   pip install adafruit-circuitpython-bno08x adafruit-blinka
   ```

---

## Camera calibration (do this now, and again with the Bolt)

```bash
python -m walker_gait.calibration.calibrate_camera \
    --images "calib_images/*.jpg" \
    --board-cols 9 --board-rows 6 --square-size-m 0.025 \
    --out configs/camera_intrinsics_measured.yaml
```

Print a standard 9x6-inner-corner checkerboard, take ~15-20 photos of it from
varied angles/distances with your current webcam, and run the above. Check
the printed mean reprojection error — under ~0.5 px is a good calibration,
over ~1.0 px means re-shoot with more varied angles. Re-run this *exact*
script, unchanged, against the Femto Bolt's RGB stream once it arrives.

---

## What to build/validate next, in order

1. Done in this repo: `SyntheticSource` + calibration script
2. Done in this repo: `IouTracker`
3. Done in this repo: `Detector` (`dummy`/`motion` today, `yolov8` stubbed)
4. Done in this repo: `DepthBackprojector` (tested against synthetic ground truth)
5. Done in this repo: `KalmanSkeletonSmoother` (tested against injected noise)
6. Done in this repo: `VerticalVelocityEventDetector` + `GaitMetricsCalculator`
7. Install `ultralytics`, swap `detector` backend from `dummy`/`motion` to
   `yolov8`, validate against `scripts/run_pipeline_webcam.py`
8. Install `mmpose`, swap `pose2d` backend from `dummy` to `rtmpose`, validate
   on real webcam footage against manually-annotated heel-strike frames
9. Wire BNO085 `ImuSource`, add independent event cross-check
10. Start MotionBERT integration against public 3D pose datasets (Human3.6M
    format) while waiting on the Bolt
11. When the Bolt arrives: implement `FemtoBoltSource`, re-run calibration,
    drop in real intrinsics — this should be close to a same-day integration
    since nothing upstream or downstream needs to change.
