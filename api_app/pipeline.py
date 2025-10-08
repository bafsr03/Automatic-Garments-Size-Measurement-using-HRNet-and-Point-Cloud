import os
import json
import shutil
import subprocess
from typing import Dict, Any, Tuple

import cv2
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_midas(input_image: str, out_dir: str) -> Tuple[str, str]:
    _ensure_dirs(out_dir)
    # Use existing MiDaS run.py in sibling repo if available; fallback to local import
    midas_repo = os.path.abspath(os.path.join(REPO_ROOT, "..", "MiDaS"))
    run_py = os.path.join(midas_repo, "run.py")
    if not os.path.exists(run_py):
        raise RuntimeError("MiDaS run.py not found; expected at ../MiDaS/run.py")

    # Copy image to MiDaS root and call run.py with env to override input/output
    tmp_input = os.path.join(midas_repo, "input.jpg")
    shutil.copyfile(input_image, tmp_input)

    env = os.environ.copy()
    # Force UTF-8 for MiDaS prints (avoids Windows cp1252 UnicodeEncodeError)
    env["PYTHONIOENCODING"] = "utf-8"
    # run.py saves to MiDaS/output; we will copy results back
    try:
        completed = subprocess.run([sys.executable, run_py], cwd=midas_repo, env=env, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        msg = (
            "MiDaS run.py failed.\n"
            f"returncode: {e.returncode}\n"
            f"stdout:\n{(e.stdout or '')[-4000:]}\n"
            f"stderr:\n{(e.stderr or '')[-4000:]}\n"
        )
        raise RuntimeError(msg)

    depth_png = os.path.join(midas_repo, "output", "depth.png")
    ply_path = os.path.join(midas_repo, "output", "pointcloud.ply")
    if not os.path.exists(depth_png) or not os.path.exists(ply_path):
        raise RuntimeError("MiDaS did not produce expected outputs")

    out_depth = os.path.join(out_dir, "depth.png")
    out_ply = os.path.join(out_dir, "pointcloud.ply")
    shutil.copyfile(depth_png, out_depth)
    shutil.copyfile(ply_path, out_ply)
    return out_depth, out_ply


def build_single_image_df2_dataset(input_image: str, dataset_root: str) -> str:
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
                "category_id": 1,
                "bbox": [1, 1, max(2, w - 2), max(2, h - 2)],
                "area": float(w * h),
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
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
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)

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


def run_measurement_vis(results_json: str, ann_json: str, images_dir: str, out_dir: str) -> Tuple[str, str]:
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
        "--unit", "px",
        "--only-image-id", "1",
    ]
    env = os.environ.copy()
    try:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        msg = (
            "Measurement script failed.\n"
            f"returncode: {e.returncode}\n"
            f"stdout:\n{(e.stdout or '')[-4000:]}\n"
            f"stderr:\n{(e.stderr or '')[-4000:]}\n"
        )
        raise RuntimeError(msg)
    # Expect one image output with the same name
    vis_image = os.path.join(vis_dir, "000001.jpg")
    return vis_image, report_json


def process_image_request(input_image_path: str, work_dir: str) -> Dict[str, Any]:
    # 1) Copy original
    img_copy = os.path.join(work_dir, "image.jpg")
    if not os.path.exists(img_copy):
        shutil.copyfile(input_image_path, img_copy)

    # 2) MiDaS
    depth_path, ply_path = run_midas(img_copy, work_dir)

    # 3) Build DF2 dataset
    dataset_root = os.path.join(work_dir, "df2_dataset")
    ann_json = build_single_image_df2_dataset(img_copy, dataset_root)

    # 4) HRNet inference
    hrnet_out_dir = os.path.join(work_dir, "hrnet_out")
    results_json = run_hrnet(dataset_root, hrnet_out_dir)

    # 5) Measurement visualization (tolerate failures)
    vis_image = None
    report_json = None
    measurement_error = None
    try:
        vis_image, report_json = run_measurement_vis(
            results_json,
            ann_json,
            os.path.join(dataset_root, "validation", "image"),
            work_dir,
        )
    except Exception as e:
        measurement_error = str(e)

    result = {
        "request_dir": work_dir,
        "rgb": img_copy,
        "depth": depth_path,
        "pointcloud": ply_path,
        "annotations": ann_json,
        "keypoints_results": results_json,
    }
    if vis_image:
        result["measurement_vis"] = vis_image
    if report_json:
        result["measurement_report"] = report_json
    if measurement_error:
        result["measurement_error"] = measurement_error
    return result


