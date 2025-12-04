import os
import json
import shutil
import subprocess
from typing import Dict, Any, Tuple, Optional

import cv2
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# MiDaS removed per user request


def build_single_image_df2_dataset(input_image: str, dataset_root: str, category_id: int) -> str:
    # Expected structure: root/validation/image/000001.jpg + val-coco_style.json
    img_dir = os.path.join(dataset_root, "validation", "image")
    _ensure_dirs(img_dir)
    # standard name id=1
    file_name = "000001.jpg"
    dst_img = os.path.join(img_dir, file_name)
    shutil.copyfile(input_image, dst_img)
    # Write minimal COCO-style with single image and a single detection bbox
    img_cv = cv2.imread(dst_img)
    if img_cv is None:
        raise RuntimeError(f"Failed to read copied image at {dst_img}")
    h, w = int(img_cv.shape[0]), int(img_cv.shape[1])
    keypoints = [0] * (294 * 3)
    # Set first keypoint triple to make dataset accept this item
    keypoints[0] = 1
    keypoints[1] = 1
    keypoints[2] = 2
    ann = {
        "images": [{"file_name": file_name, "id": 1, "width": w, "height": h}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": int(category_id),
                "bbox": [1, 1, max(2, w - 2), max(2, h - 2)],
                "area": float(w * h),
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": 0,
            }
        ],
        "categories": [{"id": int(category_id), "name": "garment"}],
    }
    ann_path = os.path.join(dataset_root, "validation", "val-coco_style.json")
    with open(ann_path, "w") as f:
        json.dump(ann, f)

    # Detection results file if using detector path (optional). We'll use USE_GT_BBOX True by default.
    return ann_path


def run_hrnet(dataset_root: str, output_dir: str) -> str:
    _ensure_dirs(output_dir)
    tools_dir = os.path.join(REPO_ROOT, "tools")
    cfg_path = os.path.join(REPO_ROOT, "experiments", "deepfashion2", "hrnet", "w48_384x288_adam_lr1e-3_MeasMdl.yaml")
    model_file = os.path.join(REPO_ROOT, "models", "pose_hrnet_point-detector.pth")
    if not os.path.exists(model_file):
        raise RuntimeError("HRNet model weights not found at models/pose_hrnet_point-detector.pth")

    cmd = [
        sys.executable, os.path.join(tools_dir, "test.py"),
        "--cfg", cfg_path,
        "TEST.MODEL_FILE", model_file,
        "TEST.USE_GT_BBOX", "True",
        "DATASET.ROOT", dataset_root,
        "DATASET.TEST_SET", "validation",
        "OUTPUT_DIR", output_dir,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    # Run without raising immediately; we only need the results JSON
    # Impose a hard timeout to avoid runaway HRNet inference in production (default 90s)
    timeout_s = int(os.getenv("HRNET_TIMEOUT_SECONDS", "90"))
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as te:
        raise RuntimeError(f"HRNet timed out after {timeout_s}s; partial stdout: {(te.stdout or '')[-2000:]} stderr: {(te.stderr or '')[-2000:]} ")

    # Locate results json
    # output/deepfashion2/pose_hrnet/<cfg_name>/<time>/results/keypoints_validation_results_0.json
    results_json = None
    out_root = os.path.join(output_dir, "deepfashion2", "pose_hrnet")
    if os.path.exists(out_root):
        for root, dirs, files in os.walk(out_root):
            for fn in files:
                if fn.startswith("keypoints_validation_results_") and fn.endswith(".json"):
                    results_json = os.path.join(root, fn)
                    break
            if results_json:
                break
    if not results_json:
        # If subprocess failed, surface logs; else generic
        if proc.returncode != 0:
            msg = (
                "HRNet test.py failed and no results JSON found.\n"
                f"returncode: {proc.returncode}\n"
                f"stdout:\n{(proc.stdout or '')[-4000:]}\n"
                f"stderr:\n{(proc.stderr or '')[-4000:]}\n"
            )
            raise RuntimeError(msg)
        raise RuntimeError("Could not find HRNet keypoints results JSON")
    return results_json
def _override_results_category(results_json: str, category_id: int) -> str:
    """Write a copy of results with all category_id fields set to provided value."""
    try:
        with open(results_json, 'r') as f:
            data = json.load(f)
        for item in data:
            item['category_id'] = int(category_id)
        out_path = os.path.splitext(results_json)[0] + f"_cat{int(category_id)}.json"
        with open(out_path, 'w') as f:
            json.dump(data, f)
        return out_path
    except Exception:
        return results_json


def run_measurement_vis(results_json: str, ann_json: str, images_dir: str, out_dir: str, unit: str, px_per_cm: Optional[float]) -> Tuple[str, str]:
    _ensure_dirs(out_dir)
    vis_dir = os.path.join(out_dir, "measure_vis_keypoints")
    report_json = os.path.join(out_dir, "measure_report_keypoints.json")
    tools_dir = os.path.join(REPO_ROOT, "tools")
    script = os.path.join(tools_dir, "measure_and_visualize_keypoints.py")
    cmd = [
        sys.executable, script,
        "--results", results_json,
        "--ann", ann_json,
        "--images-dir", images_dir,
        "--out-vis-dir", vis_dir,
        "--out-report", report_json,
        "--unit", unit,
        "--only-image-id", "1",
    ]
    if unit != "px" and px_per_cm is not None:
        cmd.extend(["--px-per-cm", str(px_per_cm)])
    env = os.environ.copy()
    try:
        timeout_s = int(os.getenv("MEASURE_TIMEOUT_SECONDS", "60"))
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True, timeout=timeout_s)
    except subprocess.CalledProcessError as e:
        msg = (
            "Measurement script failed.\n"
            f"returncode: {e.returncode}\n"
            f"stdout:\n{(e.stdout or '')[-4000:]}\n"
            f"stderr:\n{(e.stderr or '')[-4000:]}\n"
        )
        raise RuntimeError(msg)
    except subprocess.TimeoutExpired as te:
        raise RuntimeError(f"Measurement script timed out; partial stdout: {(te.stdout or '')[-2000:]} stderr: {(te.stderr or '')[-2000:]}")
    # Expect one image output with the same name
    vis_image = os.path.join(vis_dir, "000001.jpg")
    return vis_image, report_json


def _compute_px_per_cm(results_json: str, category_id: int, true_waist: float) -> Tuple[Optional[float], Optional[float]]:
    """Compute scale (px/cm) using the provided waist_cm and predicted keypoints.
    Returns (px_per_cm, waist_px). If cannot compute, returns (None, None).
    """
    if true_waist is None or true_waist <= 0:
        return None, None
    import importlib
    import sys as _sys
    if REPO_ROOT not in _sys.path:
        _sys.path.insert(0, REPO_ROOT)
    try:
        mav = importlib.import_module("tools.measure_and_visualize_keypoints")
    except Exception:
        return None, None
    try:
        with open(results_json, "r") as f:
            results = json.load(f)
        if not results:
            return None, None
        # Prefer image_id == 1
        item = None
        for it in results:
            if int(it.get("image_id", -1)) == 1:
                item = it
                break
        if item is None:
            item = results[0]
        kpts = item.get("keypoints")
        if not isinstance(kpts, list):
            return None, None
        measurements = mav.compute_measurements_for_item(kpts, int(category_id), unit='px', px_per_cm=None)
        waist_px = None
        for key in ("waist", "waist_width"):
            if key in measurements:
                waist_px = float(measurements[key])
                break
        if waist_px is None or waist_px <= 0:
            return None, None
        return (waist_px / float(true_waist)), waist_px
    except Exception:
        return None, None


def _build_size_scale(measurements: Dict[str, Any], true_size: str, input_unit: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    size_order = ["XS", "S", "M", "L", "XL", "XXL"]
    size_upper = (true_size or "L").strip().upper()
    if size_upper not in size_order:
        size_upper = "L"
    base_idx = size_order.index(size_upper)

    # Prepare base measurements in both units
    base_cm = {}
    base_in = {}
    
    is_px = input_unit.lower() == 'px'
    is_inch = input_unit.lower() in ("inch", "inches", "in")
    
    # First pass: normalize input to CM and Inch
    temp_cm = {}
    temp_in = {}
    
    for k, v in measurements.items():
        if k.startswith("_") or k.endswith("_line"): continue
        if not isinstance(v, (int, float)): continue
        
        val = float(v)
        if is_px:
            temp_cm[k] = val
            temp_in[k] = val
        elif is_inch:
            temp_in[k] = val
            temp_cm[k] = val * 2.54
        else: # CM
            temp_cm[k] = val
            temp_in[k] = val / 2.54

    # Detect Half-Width (Flat) vs Girth (Circumference)
    # Heuristic: If Chest/Bust < 70cm (or < 28in), it's likely Half-Width.
    # We MUST store Girth for the recommender to work simply.
    
    chest_val = temp_cm.get("chest") or temp_cm.get("bust") or temp_cm.get("chest_width")
    is_half_width = False
    if chest_val and chest_val < 70.0:
        is_half_width = True
        print(f"[PIPELINE] Detected Half-Width measurements (Chest {chest_val:.1f}cm). Converting to Girth.")

    horizontal_keys = {"chest", "bust", "waist", "hips", "hem", "thigh", "chest_width", "bust_width", "waist_width", "hip_width"}
    
    for k in temp_cm:
        val_cm = temp_cm[k]
        val_in = temp_in[k]
        
        # Normalize key name
        k_norm = k.lower()
        if k_norm == "shoulder_to_shoulder": k_norm = "shoulder_width"
        
        # Apply doubling if half-width and is a horizontal girth measure
        # Note: Shoulder width is NOT a girth measure, so we exclude it.
        if is_half_width and any(hk in k_norm for hk in horizontal_keys) and "shoulder" not in k_norm:
            val_cm *= 2.0
            val_in *= 2.0
            
        base_cm[k_norm] = val_cm
        base_in[k_norm] = val_in

    def is_width_key(k: str) -> bool:
        k = k.lower()
        return any(w in k for w in ["width", "waist", "hip", "chest", "shoulder", "hem", "leg_opening", "thigh", "knee"]) and not k.endswith("_line")

    def is_length_key(k: str) -> bool:
        k = k.lower()
        return any(w in k for w in ["length", "inseam", "front_rise", "back_rise"]) and not k.endswith("_line")

    # Define increments
    # CM
    cm_width_inc = 4.0
    cm_length_inc = 2.0
    # Inch
    in_width_inc = 1.6
    in_length_inc = 0.8
    # PX (Percentage)
    px_width_pct = 0.05
    px_length_pct = 0.025

    scale_cm: Dict[str, Dict[str, float]] = {s: {} for s in size_order}
    scale_in: Dict[str, Dict[str, float]] = {s: {} for s in size_order}

    # Generate scales
    for i, size in enumerate(size_order):
        step = i - base_idx
        
        # CM Generation
        for key, base_val in base_cm.items():
            if is_px:
                if is_width_key(key): val = base_val * (1.0 + px_width_pct * step)
                elif is_length_key(key): val = base_val * (1.0 + px_length_pct * step)
                else: val = base_val
            else:
                if is_width_key(key): val = base_val + cm_width_inc * step
                elif is_length_key(key): val = base_val + cm_length_inc * step
                else: val = base_val
            # Round to 2 decimals for CM, whole numbers for PX
            if is_px:
                scale_cm[size][key] = float(round(max(0.0, float(val)), 0))
            else:
                scale_cm[size][key] = float(round(max(0.0, float(val)), 2))
            
        # Inch Generation
        for key, base_val in base_in.items():
            if is_px:
                if is_width_key(key): val = base_val * (1.0 + px_width_pct * step)
                elif is_length_key(key): val = base_val * (1.0 + px_length_pct * step)
                else: val = base_val
            else:
                if is_width_key(key): val = base_val + in_width_inc * step
                elif is_length_key(key): val = base_val + in_length_inc * step
                else: val = base_val
            # Round to 2 decimals for Inches, whole numbers for PX
            if is_px:
                scale_in[size][key] = float(round(max(0.0, float(val)), 0))
            else:
                scale_in[size][key] = float(round(max(0.0, float(val)), 2))


    return scale_cm, scale_in


def process_image_request(input_image_path: str, work_dir: str, category_id: int, true_size: str, true_waist: Optional[float], unit: str) -> Dict[str, Any]:
    # 1) Copy original
    img_copy = os.path.join(work_dir, "image.jpg")
    if not os.path.exists(img_copy):
        shutil.copyfile(input_image_path, img_copy)

    # 2) Depth/pointcloud removed
    depth_path: Optional[str] = None
    ply_path: Optional[str] = None

    # 3) Build DF2 dataset
    dataset_root = os.path.join(work_dir, "df2_dataset")
    ann_json = build_single_image_df2_dataset(img_copy, dataset_root, category_id)

    # 4) HRNet inference
    hrnet_out_dir = os.path.join(work_dir, "hrnet_out")
    results_json = run_hrnet(dataset_root, hrnet_out_dir)
    # Force the provided category_id in results for downstream measurement
    results_json = _override_results_category(results_json, category_id)

    # 5) Compute scale and run measurement visualization
    vis_image = None
    report_json = None
    measurement_error = None
    try:
        px_per_cm, waist_px = _compute_px_per_cm(results_json, category_id, true_waist) if true_waist is not None else (None, None)
        print(f"[PIPELINE] Running measurement visualization with px_per_cm={px_per_cm}")
        vis_image, report_json = run_measurement_vis(
            results_json,
            ann_json,
            os.path.join(dataset_root, "validation", "image"),
            work_dir,
            unit=unit,
            px_per_cm=px_per_cm,
        )
        print(f"[PIPELINE] Measurement visualization completed: {vis_image}")
    except Exception as e:
        print(f"[PIPELINE] Measurement visualization failed: {e}")
        measurement_error = str(e)

    # Build size scale JSON based on report measurements of this single image
    size_scale_path = os.path.join(work_dir, "size_scale.json")
    try:
        measurements_single: Dict[str, Any] = {}
        if report_json and os.path.exists(report_json):
            with open(report_json, 'r') as f:
                rep = json.load(f)
            if isinstance(rep, list) and len(rep) > 0:
                measurements_single = rep[0].get('measurements', {})
        if measurements_single:
            scale_cm, scale_in = _build_size_scale(measurements_single, true_size=true_size, input_unit=unit)
            with open(size_scale_path, 'w') as f:
                json.dump({
                    "units": ["cm", "inch"],
                    "true_size": true_size,
                    "scale_cm": scale_cm,
                    "scale_in": scale_in,
                    # Legacy support: default to input unit
                    "unit": unit,
                    "scale": scale_in if unit.lower() in ("inch", "inches", "in") else scale_cm
                }, f, indent=2)
        else:
            # still write an empty scale file for consistency
            with open(size_scale_path, 'w') as f:
                json.dump({"units": ["cm", "inch"], "true_size": true_size, "scale_cm": {}, "scale_in": {}, "unit": unit, "scale": {}}, f, indent=2)
    except Exception as e:
        measurement_error = (str(e) if measurement_error is None else measurement_error)

    # Respond minimally per request: only visualization image and size scale JSON
    out: Dict[str, Any] = {}
    # Return relative paths from base directory for file serving
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if vis_image:
        rel_vis_path = os.path.relpath(vis_image, base_dir)
        out["measurement_vis"] = rel_vis_path
    rel_size_scale_path = os.path.relpath(size_scale_path, base_dir)
    out["size_scale"] = rel_size_scale_path
    # Include error if any
    if measurement_error:
        out["error"] = measurement_error
    return out


