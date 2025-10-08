## Garment Measurement API (FastAPI)

This adds a simple HTTP API to orchestrate MiDaS depth, point cloud generation, HRNet keypoints, and measurement visualization for a single uploaded image.

### Endpoints

- POST `/process` (multipart/form-data): field `image`.
  - Returns JSON with absolute paths to:
    - `rgb`, `depth`, `pointcloud`, `annotations` (val-coco_style.json),
    - `keypoints_results` (HRNet results JSON),
    - `measurement_vis` (annotated image),
    - `measurement_report` (JSON of measurements)

- GET `/files?path=ABSOLUTE_PATH` to download any produced file.

### Run locally

Requirements in addition to repo requirements: `fastapi`, `uvicorn`.

```bash
pip install fastapi uvicorn

# From repo root
uvicorn api_app.main:app --host 0.0.0.0 --port 8000
```

Then POST an image:

```bash
curl -F "image=@/path/to/your.jpg" http://localhost:8000/process
```

Notes:
- Expects MiDaS sibling repo at `../MiDaS` with `run.py` and weights.
- Uses existing HRNet config `experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3_MeasMdl.yaml` and weights at `models/pose_hrnet_point-detector.pth`.
- Creates a temporary single-image DeepFashion2-style dataset under `api_runs/<uuid>/df2_dataset/validation/`.


