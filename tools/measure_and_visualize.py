import os
import json
import math
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from pycocotools.coco import COCO


# DeepFashion2 category id to (start, end) index of keypoints inside the 294-length array (1-based ranges in paper -> here 0-based half-open)
DF2_CATEGORY_KP_RANGES: Dict[int, Tuple[int, int]] = {
    1: (0, 25),
    2: (25, 58),
    3: (58, 89),
    4: (89, 128),
    5: (128, 143),
    6: (143, 158),
    7: (158, 168),
    8: (168, 182),
    9: (182, 190),
    10: (190, 219),
    11: (219, 256),
    12: (256, 275),
    13: (275, 294),
}


# Category groups for measurement presets
TOPS = {1, 2, 3, 4, 5, 6}
BOTTOMS_SHORTS = {7}
BOTTOMS_TROUSERS = {8}
SKIRTS = {9}
DRESSES = {10, 11, 12, 13}


def _extract_category_keypoints(all_kpts_flat: List[float], category_id: int) -> np.ndarray:
    """
    Convert the 294*3 flat list into an array [N, 3] for the specific category.
    Returns Nx3 array [x, y, score]. Filters nothing yet.
    """
    start, end = DF2_CATEGORY_KP_RANGES[category_id]
    num = end - start
    # all_kpts_flat is [x1, y1, s1, x2, y2, s2, ...] length 294*3
    kpts = np.array(all_kpts_flat, dtype=np.float32).reshape(-1, 3)  # [294, 3]
    return kpts[start:end, :]  # [num, 3]


def _visible_points(kpts_cat: np.ndarray, score_thr: float = 0.2) -> np.ndarray:
    xs = kpts_cat[:, 0]
    ys = kpts_cat[:, 1]
    ss = kpts_cat[:, 2]
    mask = (ss >= score_thr) & (xs >= 0) & (ys >= 0)
    return kpts_cat[mask, :2]


def _convex_hull(points_xy: np.ndarray) -> Optional[np.ndarray]:
    if points_xy.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(points_xy)
        return points_xy[hull.vertices]
    except Exception:
        return None


def _y_fraction_line_width(hull_xy: np.ndarray, frac: float) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Given convex hull points in order (may be unordered; we'll order by cv2.convexHull), compute the max width at y = y_min + frac*(y_max-y_min).
    Returns ((x_left, y), (x_right, y)) or None if cannot compute.
    """
    if hull_xy is None or len(hull_xy) < 3:
        return None
    # Order hull points for robust edge traversal
    hull_xy_ordered = cv2.convexHull(hull_xy.astype(np.float32)).squeeze(1)  # [M,2]
    y_min = float(hull_xy_ordered[:, 1].min())
    y_max = float(hull_xy_ordered[:, 1].max())
    if y_max <= y_min:
        return None
    y_target = y_min + frac * (y_max - y_min)

    xs: List[float] = []
    M = hull_xy_ordered.shape[0]
    for i in range(M):
        x1, y1 = hull_xy_ordered[i]
        x2, y2 = hull_xy_ordered[(i + 1) % M]
        # Check if horizontal line at y_target intersects edge (y between y1 and y2)
        if (y1 <= y_target <= y2) or (y2 <= y_target <= y1):
            if y2 == y1:
                # Horizontal edge: take both x's
                xs.extend([float(x1), float(x2)])
            else:
                t = (y_target - y1) / (y2 - y1)
                x_at = x1 + t * (x2 - x1)
                xs.append(float(x_at))

    if len(xs) < 2:
        return None
    xs_sorted = sorted(xs)
    # Pair them; choose the widest span
    max_w = -1.0
    best_pair = None
    for i in range(0, len(xs_sorted) - 1, 2):
        w = xs_sorted[i + 1] - xs_sorted[i]
        if w > max_w:
            max_w = w
            best_pair = (xs_sorted[i], xs_sorted[i + 1])

    if best_pair is None:
        return None
    return (best_pair[0], y_target), (best_pair[1], y_target)


def _bbox_length(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] == 0:
        return 0.0
    y_min = float(points_xy[:, 1].min())
    y_max = float(points_xy[:, 1].max())
    return max(0.0, y_max - y_min)


def _convert_units(px: float, unit: str, px_per_cm: Optional[float]) -> Tuple[float, str]:
    unit = unit.lower()
    if unit == 'px' or px_per_cm is None:
        return px, 'px'
    if unit in ('cm', 'centimeter', 'centimeters'):
        return px / px_per_cm, 'cm'
    if unit in ('inch', 'inches', 'in'):
        return px / (px_per_cm * 2.54), 'in'
    # default fallback
    return px, 'px'


def compute_measurements_for_item(kpts_flat: List[float], category_id: int, unit: str, px_per_cm: Optional[float]) -> Dict[str, float]:
    kpts_cat = _extract_category_keypoints(kpts_flat, category_id)
    pts = _visible_points(kpts_cat)
    hull = _convex_hull(pts)
    measurements: Dict[str, float] = {}
    if hull is None:
        # Not enough points; return empty
        return measurements

    # Common measurements based on fractions of height
    length_px = _bbox_length(pts)
    length_val, unit_lbl = _convert_units(length_px, unit, px_per_cm)

    if category_id in TOPS or category_id in DRESSES:
        # Fractions tuned for upper-body garments
        pairs = {
            'shoulder_width': 0.15,
            'chest_width': 0.35,
            'waist_width': 0.55,
            'hem_width': 0.90,
        }
        for name, frac in pairs.items():
            pair = _y_fraction_line_width(hull, frac)
            if pair is not None:
                (xl, y), (xr, _) = pair
                val_px = max(0.0, xr - xl)
                val, _ = _convert_units(val_px, unit, px_per_cm)
                measurements[name] = float(val)
        # Body length
        measurements['body_length'] = float(length_val)

    if category_id in BOTTOMS_TROUSERS or category_id in BOTTOMS_SHORTS or category_id in SKIRTS:
        pairs = {
            'waist_width': 0.10,
            'hip_width': 0.35,
            'thigh_width': 0.50,
            'hem_width': 0.95,
        }
        for name, frac in pairs.items():
            pair = _y_fraction_line_width(hull, frac)
            if pair is not None:
                (xl, y), (xr, _) = pair
                val_px = max(0.0, xr - xl)
                val, _ = _convert_units(val_px, unit, px_per_cm)
                measurements[name] = float(val)
        # Outseam length (overall height)
        measurements['outseam_length'] = float(length_val)

    # Add unit label once
    measurements['_unit'] = unit_lbl
    return measurements


def _draw_line_with_label(img: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float], label: str, color: Tuple[int, int, int]) -> None:
    p1i = (int(round(p1[0])), int(round(p1[1])))
    p2i = (int(round(p2[0])), int(round(p2[1])))
    cv2.line(img, p1i, p2i, color, 2, cv2.LINE_AA)
    mid = (int((p1i[0] + p2i[0]) / 2), int((p1i[1] + p2i[1]) / 2))
    cv2.putText(img, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def visualize_item(image_path: str,
                   kpts_flat: List[float],
                   category_id: int,
                   unit: str,
                   px_per_cm: Optional[float],
                   out_path: str) -> Dict[str, float]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    kpts_cat = _extract_category_keypoints(kpts_flat, category_id)
    pts = _visible_points(kpts_cat)
    # Draw keypoints
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1, cv2.LINE_AA)

    hull = _convex_hull(pts)
    if hull is not None and hull.shape[0] >= 3:
        hull_ordered = cv2.convexHull(hull.astype(np.float32)).squeeze(1)
        cv2.polylines(img, [hull_ordered.astype(np.int32)], True, (255, 200, 0), 1, cv2.LINE_AA)

    measurements = compute_measurements_for_item(kpts_flat, category_id, unit, px_per_cm)
    unit_lbl = measurements.get('_unit', unit)

    # Draw measurement lines used
    # Tops/dresses
    colors = {
        'shoulder_width': (0, 165, 255),  # orange
        'chest_width': (255, 0, 0),       # blue
        'waist_width': (0, 0, 255),       # red
        'hip_width': (255, 0, 255),       # magenta
        'hem_width': (0, 255, 255),       # yellow
        'body_length': (0, 255, 0),       # green
        'outseam_length': (0, 255, 0),
        'thigh_width': (128, 0, 128),
    }

    if hull is not None:
        if category_id in TOPS or category_id in DRESSES:
            for name, frac in [('shoulder_width', 0.15), ('chest_width', 0.35), ('waist_width', 0.55), ('hem_width', 0.90)]:
                pair = _y_fraction_line_width(hull, frac)
                if pair is not None and name in measurements:
                    (xl, y), (xr, _) = pair
                    length = measurements[name]
                    _draw_line_with_label(img, (xl, y), (xr, y), f"{name}: {length:.1f} {unit_lbl}", colors.get(name, (255, 255, 255)))

            # vertical length line at center x
            y_min = float(pts[:, 1].min()) if pts.shape[0] else 0.0
            y_max = float(pts[:, 1].max()) if pts.shape[0] else 0.0
            x_center = float(pts[:, 0].mean()) if pts.shape[0] else 0.0
            if 'body_length' in measurements:
                _draw_line_with_label(img, (x_center, y_min), (x_center, y_max), f"body_length: {measurements['body_length']:.1f} {unit_lbl}", colors['body_length'])

        if category_id in BOTTOMS_TROUSERS or category_id in BOTTOMS_SHORTS or category_id in SKIRTS:
            for name, frac in [('waist_width', 0.10), ('hip_width', 0.35), ('thigh_width', 0.50), ('hem_width', 0.95)]:
                pair = _y_fraction_line_width(hull, frac)
                if pair is not None and name in measurements:
                    (xl, y), (xr, _) = pair
                    length = measurements[name]
                    _draw_line_with_label(img, (xl, y), (xr, y), f"{name}: {length:.1f} {unit_lbl}", colors.get(name, (255, 255, 255)))

            # vertical length
            y_min = float(pts[:, 1].min()) if pts.shape[0] else 0.0
            y_max = float(pts[:, 1].max()) if pts.shape[0] else 0.0
            x_center = float(pts[:, 0].mean()) if pts.shape[0] else 0.0
            if 'outseam_length' in measurements:
                _draw_line_with_label(img, (x_center, y_min), (x_center, y_max), f"outseam_length: {measurements['outseam_length']:.1f} {unit_lbl}", colors['outseam_length'])

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return measurements


def run_from_results(results_json: str,
                     ann_json: str,
                     images_dir: str,
                     out_vis_dir: str,
                     out_report_json: str,
                     unit: str = 'cm',
                     px_per_cm: Optional[float] = None,
                     score_thr: float = 0.2,
                     limit: Optional[int] = None) -> None:
    with open(results_json, 'r') as f:
        results = json.load(f)

    coco = COCO(ann_json)
    cat_id_to_name = {c['id']: c['name'] for c in coco.loadCats(coco.getCatIds())}
    imgid_to_file = {}
    for img in coco.loadImgs(coco.getImgIds()):
        imgid_to_file[img['id']] = img['file_name']

    report: List[Dict] = []
    count = 0
    for item in results:
        if limit is not None and count >= limit:
            break
        img_id = item['image_id']
        category_id = int(item['category_id']) if isinstance(item['category_id'], (int, float)) else int(float(item['category_id']))
        kpts = item['keypoints']
        fname = imgid_to_file.get(img_id)
        if not fname:
            continue
        image_path = os.path.join(images_dir, fname)
        out_path = os.path.join(out_vis_dir, fname)
        try:
            measurements = visualize_item(image_path, kpts, category_id, unit, px_per_cm, out_path)
        except Exception as e:
            # Skip problematic item but continue
            continue
        entry = {
            'image_id': img_id,
            'file_name': fname,
            'category_id': category_id,
            'category_name': cat_id_to_name.get(category_id, str(category_id)),
            'measurements': measurements,
        }
        report.append(entry)
        count += 1

    os.makedirs(os.path.dirname(out_report_json), exist_ok=True)
    with open(out_report_json, 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DeepFashion2 measurement and visualization from results JSON')
    parser.add_argument('--results', required=True, help='Path to keypoints results JSON (from tools/test.py)')
    parser.add_argument('--ann', required=True, help='Path to annotation JSON (val-coco_style.json)')
    parser.add_argument('--images-dir', required=True, help='Path to validation images directory')
    parser.add_argument('--out-vis-dir', default='output/measure_vis', help='Directory to save visualizations')
    parser.add_argument('--out-report', default='output/measure_report.json', help='Output JSON with measurements')
    parser.add_argument('--unit', default='cm', choices=['cm', 'inch', 'px'])
    parser.add_argument('--px-per-cm', type=float, default=None, help='Pixels per centimeter (required for cm/inch)')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Keypoint score threshold for visibility')
    parser.add_argument('--limit', type=int, default=None, help='Optional, limit number of items to process')

    args = parser.parse_args()

    if args.unit != 'px' and args.px_per_cm is None:
        print('[WARN] --px-per-cm is required for unit cm/inch. Falling back to px in outputs.')

    run_from_results(
        results_json=args.results,
        ann_json=args.ann,
        images_dir=args.images_dir,
        out_vis_dir=args.out_vis_dir,
        out_report_json=args.out_report,
        unit=args.unit,
        px_per_cm=args.px_per_cm,
        score_thr=args.score_thr,
        limit=args.limit,
    )


