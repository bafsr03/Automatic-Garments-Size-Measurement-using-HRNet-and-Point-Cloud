import os
import json
import math
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
# Optional dependency: pycocotools. Provide a simple fallback reader on Windows.
try:
    from pycocotools.coco import COCO as _PY_COCO  # type: ignore
    def _load_coco(ann_path: str):
        return _PY_COCO(ann_path)
except Exception:
    # Minimal COCO-like helper used by this script only
    class _SimpleCOCO:
        def __init__(self, ann_path: str):
            with open(ann_path, 'r') as f:
                data = json.load(f)
            self._images = {int(img['id']): img for img in data.get('images', [])}
            self._categories = {int(cat['id']): cat for cat in data.get('categories', [])}

        def getCatIds(self) -> List[int]:
            return list(self._categories.keys())

        def loadCats(self, cat_ids: List[int]) -> List[Dict[str, Any]]:
            return [self._categories[cid] for cid in cat_ids if cid in self._categories]

        def getImgIds(self) -> List[int]:
            return list(self._images.keys())

        def loadImgs(self, img_ids: List[int]) -> List[Dict[str, Any]]:
            return [self._images[iid] for iid in img_ids if iid in self._images]

    def _load_coco(ann_path: str):
        return _SimpleCOCO(ann_path)


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

# Category groups for measurement presets (per user request)
# Supported categories only; others will be skipped
GROUP_1_SHORT_SLEEVE_TOP_OUTWEAR = {1, 3}
GROUP_2_LONG_SLEEVE_TOP_OUTWEAR = {2, 4}
GROUP_3_VEST = {5}
GROUP_4_SHORTS = {7}
GROUP_5_TROUSERS = {8}
GROUP_9_SKIRT = {9}
GROUP_10_SHORT_SLEEVE_DRESS = {10}
GROUP_11_LONG_SLEEVE_DRESS = {11}

SUPPORTED_CATEGORIES = (
    GROUP_1_SHORT_SLEEVE_TOP_OUTWEAR
    | GROUP_2_LONG_SLEEVE_TOP_OUTWEAR
    | GROUP_3_VEST
    | GROUP_4_SHORTS
    | GROUP_5_TROUSERS
    | GROUP_9_SKIRT
    | GROUP_10_SHORT_SLEEVE_DRESS
    | GROUP_11_LONG_SLEEVE_DRESS
)

def _category_group(category_id: int) -> Optional[str]:
    if category_id in GROUP_1_SHORT_SLEEVE_TOP_OUTWEAR:
        return 'G1_SHORT_TOP_OUTWEAR'
    if category_id in GROUP_2_LONG_SLEEVE_TOP_OUTWEAR:
        return 'G2_LONG_TOP_OUTWEAR'
    if category_id in GROUP_3_VEST:
        return 'G3_VEST'
    if category_id in GROUP_4_SHORTS:
        return 'G4_SHORTS'
    if category_id in GROUP_5_TROUSERS:
        return 'G5_TROUSERS'
    if category_id in GROUP_9_SKIRT:
        return 'G9_SKIRT'
    if category_id in GROUP_10_SHORT_SLEEVE_DRESS:
        return 'G10_SHORT_DRESS'
    if category_id in GROUP_11_LONG_SLEEVE_DRESS:
        return 'G11_LONG_DRESS'
    return None

# Edit the following mapping to change which keypoint segments are measured
# per group. Indices are 1-based within the category's own keypoint set.
# type: 'length' uses Euclidean distance; 'width' draws a horizontal line.
KP_SEGMENTS_BY_GROUP = {
    'G1_SHORT_TOP_OUTWEAR': [
        {'name': 'neck', 'type': 'length', 'a': 2, 'b': 6},
        {'name': 'shoulder_to_shoulder', 'type': 'width', 'a': 7, 'b': 25},
        {'name': 'chest', 'type': 'width', 'a': 12, 'b': 20},
        {'name': 'waist', 'type': 'width', 'a': 14, 'b': 18},
        {'name': 'hem', 'type': 'width', 'a': 15, 'b': 17},
        {'name': 'sleeve_length', 'type': 'length', 'a': 25, 'b': 23},
        {'name': 'sleeve', 'type': 'length', 'a': 23, 'b': 22},
    ],
    'G2_LONG_TOP_OUTWEAR': [
        {'name': 'neck', 'type': 'length', 'a': 2, 'b': 6},
        {'name': 'shoulder_to_shoulder', 'type': 'width', 'a': 7, 'b': 33},
        {'name': 'chest', 'type': 'width', 'a': 17, 'b': 23},
        {'name': 'waist', 'type': 'width', 'a': 18, 'b': 22},
        {'name': 'hem', 'type': 'width', 'a': 19, 'b': 21},
        {'name': 'sleeve_length', 'type': 'length', 'a': 7, 'b': 11},
        {'name': 'sleeve', 'type': 'length', 'a': 11, 'b': 12},
    ],
    'G3_VEST': [
        {'name': 'neck', 'type': 'length', 'a': 2, 'b': 6},
        {'name': 'shoulder_to_shoulder', 'type': 'width', 'a': 7, 'b': 25},
        {'name': 'chest', 'type': 'width', 'a': 12, 'b': 20},
        {'name': 'waist', 'type': 'width', 'a': 14, 'b': 18},
        {'name': 'hem', 'type': 'width', 'a': 15, 'b': 17},
    ],
    'G10_SHORT_DRESS': [
        {'name': 'neck', 'type': 'length', 'a': 2, 'b': 6},
        {'name': 'shoulder_to_shoulder', 'type': 'width', 'a': 7, 'b': 25},
        {'name': 'chest', 'type': 'width', 'a': 12, 'b': 20},
        {'name': 'waist', 'type': 'width', 'a': 14, 'b': 18},
        {'name': 'hem', 'type': 'width', 'a': 15, 'b': 17},
        {'name': 'sleeve_length', 'type': 'length', 'a': 25, 'b': 23},
        {'name': 'sleeve', 'type': 'length', 'a': 23, 'b': 22},
    ],
    'G11_LONG_DRESS': [
        {'name': 'neck', 'type': 'length', 'a': 2, 'b': 6},
        {'name': 'shoulder_to_shoulder', 'type': 'width', 'a': 7, 'b': 25},
        {'name': 'chest', 'type': 'width', 'a': 12, 'b': 20},
        {'name': 'waist', 'type': 'width', 'a': 14, 'b': 18},
        {'name': 'hem', 'type': 'width', 'a': 15, 'b': 17},
        {'name': 'sleeve_length', 'type': 'length', 'a': 25, 'b': 23},
        {'name': 'sleeve', 'type': 'length', 'a': 23, 'b': 22},
    ],
    # Bottoms: define custom KP segments here if you want exact KP-based measures.
    # Leave lists empty to fallback to band-based widths.
    'G4_SHORTS': [
         {'name': 'waist', 'type': 'width', 'a': 1, 'b': 3},
         {'name': 'front_rise', 'type': 'length', 'a': 2, 'b': 7},
         {'name': 'back_rise', 'type': 'length', 'a': 4, 'b': 10},
         {'name': 'inseam', 'type': 'length', 'a': 4, 'b': 7},
         {'name': 'length', 'type': 'length', 'a': 1, 'b': 5},   
         {'name': 'half_knee', 'type': 'length', 'a': 7, 'b': 6},
         {'name': 'leg_opening', 'type': 'length', 'a': 5, 'b': 6},        
    ],

    'G5_TROUSERS': [
         {'name': 'waist', 'type': 'length', 'a': 3, 'b': 1},
         {'name': 'front_rise', 'type': 'width', 'a': 2, 'b': 9},
         {'name': 'hip', 'type': 'width', 'a': 15, 'b': 17},
         {'name': 'thigh', 'type': 'length', 'a': 9, 'b': 4},   
         {'name': 'length', 'type': 'length', 'a': 6, 'b': 1},   
         {'name': 'knee', 'type': 'length', 'a': 8, 'b': 5},
         {'name': 'inseam', 'type': 'length', 'a': 9, 'b': 7},
         {'name': 'leg_opening', 'type': 'length', 'a': 7, 'b': 6},
    ],
    'G9_SKIRT': [
         {'name': 'waist', 'type': 'width', 'a': 1, 'b': 3},
         {'name': 'hip', 'type': 'width', 'a': 4, 'b': 8},
         {'name': 'length', 'type': 'length', 'a': 2, 'b': 6},
         {'name': 'hem', 'type': 'length', 'a': 7, 'b': 5},   
        
    ],
}

###############################
# Band-based measurement model
###############################
# We derive horizontal widths from keypoints by looking at narrow horizontal
# bands at category-specific vertical fractions of the garment height. This
# uses only keypoints (no convex hull) and is robust across categories with
# different keypoint semantics.


def _extract_category_keypoints(all_kpts_flat: List[float], category_id: int) -> np.ndarray:
    """
    Convert the 294*3 flat list into an array [N, 3] for the specific category.
    Returns Nx3 array [x, y, score]. Filters nothing yet.
    """
    start, end = DF2_CATEGORY_KP_RANGES[category_id]
    # all_kpts_flat is [x1, y1, s1, x2, y2, s2, ...] length 294*3
    kpts = np.array(all_kpts_flat, dtype=np.float32).reshape(-1, 3)  # [294, 3]
    return kpts[start:end, :]  # [num, 3]


def _visible_points(kpts_cat: np.ndarray, score_thr: float = 0.2) -> np.ndarray:
    xs = kpts_cat[:, 0]
    ys = kpts_cat[:, 1]
    ss = kpts_cat[:, 2]
    mask = (ss >= score_thr) & (xs >= 0) & (ys >= 0)
    return kpts_cat[mask, :2]


def _band_width_from_keypoints(pts_xy: np.ndarray,
                               frac: float,
                               band_rel_height: float) -> Optional[Tuple[float, Tuple[float, float], Tuple[float, float]]]:
    """
    Compute width using only keypoints within a horizontal band at a given
    vertical fraction. Returns (width_in_px, left_point, right_point).
    - pts_xy: Nx2 array of visible (x, y)
    - frac: y fraction in [0,1] from top to bottom
    - band_rel_height: height of the band relative to full height (e.g., 0.08)
    """
    if pts_xy.shape[0] == 0:
        return None
    y_min = float(pts_xy[:, 1].min())
    y_max = float(pts_xy[:, 1].max())
    H = y_max - y_min
    if H <= 0:
        return None
    band_half = 0.5 * band_rel_height * H
    y_target = y_min + frac * H
    y_low = y_target - band_half
    y_high = y_target + band_half
    band_pts = pts_xy[(pts_xy[:, 1] >= y_low) & (pts_xy[:, 1] <= y_high)]
    # If band has too few points, relax to nearest points by y distance
    if band_pts.shape[0] < 2:
        # pick 4 nearest by |y - y_target|
        idx = np.argsort(np.abs(pts_xy[:, 1] - y_target))[:4]
        band_pts = pts_xy[idx]
        if band_pts.shape[0] < 2:
            return None
    # width by extreme x
    left_idx = np.argmin(band_pts[:, 0])
    right_idx = np.argmax(band_pts[:, 0])
    xl, yl = float(band_pts[left_idx, 0]), float(band_pts[left_idx, 1])
    xr, yr = float(band_pts[right_idx, 0]), float(band_pts[right_idx, 1])
    width_px = max(0.0, xr - xl)
    # align y to average to make line horizontal
    y_line = (yl + yr) * 0.5
    return width_px, (xl, y_line), (xr, y_line)


def _scan_band_width(
    pts_xy: np.ndarray,
    start_frac: float,
    end_frac: float,
    band_rel_height: float,
    objective: str = 'max',
    step: float = 0.01
) -> Optional[Tuple[float, Tuple[float, float], Tuple[float, float]]]:
    """
    Slide a horizontal band between [start_frac, end_frac] and pick the line
    that maximizes or minimizes width. Returns (width_px, left_pt, right_pt).
    objective: 'max' or 'min'
    """
    if pts_xy.shape[0] < 2:
        return None
    best: Optional[Tuple[float, Tuple[float, float], Tuple[float, float]]] = None
    # Ensure valid order
    s = max(0.0, min(1.0, start_frac))
    e = max(0.0, min(1.0, end_frac))
    if e < s:
        s, e = e, s
    f = s
    while f <= e + 1e-9:
        bw = _band_width_from_keypoints(pts_xy, f, band_rel_height)
        if bw is not None:
            if best is None:
                best = bw
            else:
                if objective == 'max':
                    if bw[0] > best[0]:
                        best = bw
                else:
                    if bw[0] < best[0]:
                        best = bw
        f += step
    return best


def _calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def _calculate_length_from_points(kpts_cat: np.ndarray, score_thr: float = 0.2) -> float:
    """Calculate overall length from top to bottom keypoints."""
    pts = _visible_points(kpts_cat, score_thr)
    if len(pts) < 2:
        return 0.0
    
    y_min = float(pts[:, 1].min())
    y_max = float(pts[:, 1].max())
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


def compute_measurements_for_item(kpts_flat: List[float], category_id: int, unit: str, px_per_cm: Optional[float], score_thr: float = 0.2) -> Dict[str, Any]:
    kpts_cat = _extract_category_keypoints(kpts_flat, category_id)
    pts = _visible_points(kpts_cat)
    measurements: Dict[str, Any] = {}

    # Unit label (used for all values we compute)
    unit_lbl = _convert_units(1.0, unit, px_per_cm)[1]

    # Generic vertical length (top-to-bottom span of visible keypoints)
    length_val = _calculate_length_from_points(kpts_cat, score_thr)
    # Determine supported measurement group for this category
    group_key = _category_group(category_id)
    # For tops/dresses groups, we prefer explicit front length over generic span.

    # Band fractions per garment group (top=0.0, bottom=1.0)
    # These were chosen to capture typical anatomical widths.
    if group_key in ('G1_SHORT_TOP_OUTWEAR', 'G2_LONG_TOP_OUTWEAR', 'G3_VEST', 'G10_SHORT_DRESS', 'G11_LONG_DRESS'):
        # Top-specific direct keypoint pairs (1-based indices within category)
        # Mapping provided by user
        def get_pt(idx1b: int) -> Optional[Tuple[float, float]]:
            i = idx1b - 1
            if 0 <= i < kpts_cat.shape[0]:
                x, y, s = kpts_cat[i]
                # For explicit KP pairs, be permissive: accept if coordinates look valid
                if x > -100.0 and y > -100.0:
                    return float(x), float(y)
            return None

        def add_width_from_pair(name: str, a: int, b: int):
            pa, pb = get_pt(a), get_pt(b)
            if pa is None or pb is None:
                return
            # width = horizontal delta
            width_px = abs(pb[0] - pa[0])
            val, _ = _convert_units(width_px, unit, px_per_cm)
            measurements[name] = float(val)
            measurements[f'{name}_line'] = {'left': [float(min(pa[0], pb[0])), float((pa[1]+pb[1])/2.0)], 'right': [float(max(pa[0], pb[0])), float((pa[1]+pb[1])/2.0)]}

        def add_length_from_pair(name: str, a: int, b: int):
            pa, pb = get_pt(a), get_pt(b)
            if pa is None or pb is None:
                return
            # length as Euclidean
            dist_px = math.hypot(pb[0]-pa[0], pb[1]-pa[1])
            val, _ = _convert_units(dist_px, unit, px_per_cm)
            measurements[name] = float(val)
            measurements[f'{name}_line'] = {'left': [float(pa[0]), float(pa[1])], 'right': [float(pb[0]), float(pb[1])]} 
        # Populate from centralized per-group mapping
        for seg in KP_SEGMENTS_BY_GROUP.get(group_key, []):
            name = seg['name']
            a = int(seg['a'])
            b = int(seg['b'])
            if seg['type'] == 'length':
                add_length_from_pair(name, a, b)
            else:
                add_width_from_pair(name, a, b)

        # Front length: group-specific
        # - For long sleeve top/outwear (G2), use keypoint 1 -> 20
        # - Otherwise, HPS (1) to hemline midpoint between (15,17)
        p1 = get_pt(1)
        if group_key == 'G2_LONG_TOP_OUTWEAR':
            p20 = get_pt(20)
            if p1 is not None and p20 is not None:
                dist_px = math.hypot(p20[0]-p1[0], p20[1]-p1[1])
                val, _ = _convert_units(dist_px, unit, px_per_cm)
                measurements['front_length'] = float(val)
                measurements['front_length_line'] = {'left': [float(p1[0]), float(p1[1])], 'right': [float(p20[0]), float(p20[1])]} 
        else:
            p15 = get_pt(15)
            p17 = get_pt(17)
            if p1 is not None and p15 is not None and p17 is not None:
                mid = ((p15[0]+p17[0])/2.0, (p15[1]+p17[1])/2.0)
                dist_px = math.hypot(mid[0]-p1[0], mid[1]-p1[1])
                val, _ = _convert_units(dist_px, unit, px_per_cm)
                measurements['front_length'] = float(val)
                measurements['front_length_line'] = {'left': [float(p1[0]), float(p1[1])], 'right': [float(mid[0]), float(mid[1])]} 

    if group_key in ('G4_SHORTS', 'G5_TROUSERS', 'G9_SKIRT'):
        # If KP segments are specified for this group, use them first
        segs = KP_SEGMENTS_BY_GROUP.get(group_key, [])
        used_kp_segments = False
        if len(segs) > 0:
            # Reuse helpers from tops section by defining local getters
            def get_pt(idx1b: int) -> Optional[Tuple[float, float]]:
                i = idx1b - 1
                if 0 <= i < kpts_cat.shape[0]:
                    x, y, s = kpts_cat[i]
                    if x > -100.0 and y > -100.0:
                        return float(x), float(y)
                return None
            def add_width_from_pair(name: str, a: int, b: int):
                pa, pb = get_pt(a), get_pt(b)
                if pa is None or pb is None:
                    return
                width_px = abs(pb[0] - pa[0])
                val, _ = _convert_units(width_px, unit, px_per_cm)
                measurements[name] = float(val)
                measurements[f'{name}_line'] = {'left': [float(min(pa[0], pb[0])), float((pa[1]+pb[1])/2.0)], 'right': [float(max(pa[0], pb[0])), float((pa[1]+pb[1])/2.0)]}
            def add_length_from_pair(name: str, a: int, b: int):
                pa, pb = get_pt(a), get_pt(b)
                if pa is None or pb is None:
                    return
                dist_px = math.hypot(pb[0]-pa[0], pb[1]-pa[1])
                val, _ = _convert_units(dist_px, unit, px_per_cm)
                measurements[name] = float(val)
                measurements[f'{name}_line'] = {'left': [float(pa[0]), float(pa[1])], 'right': [float(pb[0]), float(pb[1])]} 
            for seg in segs:
                name = seg['name']
                a = int(seg['a'])
                b = int(seg['b'])
                if seg['type'] == 'length':
                    add_length_from_pair(name, a, b)
                else:
                    add_width_from_pair(name, a, b)
            used_kp_segments = True

        if not used_kp_segments:
            # Fallback: band scanning widths
            band_height = 0.10
            windows = {
                'waist_width': (0.02, 0.12, 'max'),
                'hip_width': (0.20, 0.40, 'max'),
                'thigh_width': (0.45, 0.65, 'max'),
                'hem_width': (0.90, 0.99, 'max'),
            }
            for name, (f0, f1, obj) in windows.items():
                bw = _scan_band_width(pts, f0, f1, band_height, objective=obj)
                if bw is not None:
                    width_px, pL, pR = bw
                    val, _ = _convert_units(width_px, unit, px_per_cm)
                    measurements[name] = float(val)
                    measurements[f'{name}_line'] = {'left': [float(pL[0]), float(pL[1])], 'right': [float(pR[0]), float(pR[1])]}
        # Do not expose generic vertical length for bottoms (hide outseam)

    measurements['_unit'] = unit_lbl
    return measurements


def _draw_line_with_label(img: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float], label: str, color: Tuple[int, int, int]) -> None:
    p1i = (int(round(p1[0])), int(round(p1[1])))
    p2i = (int(round(p2[0])), int(round(p2[1])))
    cv2.line(img, p1i, p2i, color, 3, cv2.LINE_AA)
    mid = (int((p1i[0] + p2i[0]) / 2), int((p1i[1] + p2i[1]) / 2))
    cv2.putText(img, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def visualize_item(image_path: str,
                   kpts_flat: List[float],
                   category_id: int,
                   unit: str,
                   px_per_cm: Optional[float],
                   out_path: str,
                   score_thr: float = 0.2) -> Dict[str, float]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    kpts_cat = _extract_category_keypoints(kpts_flat, category_id)
    pts = _visible_points(kpts_cat)

    # Draw keypoints with indices (1-based per category)
    for idx, (x, y, s) in enumerate(kpts_cat):
        if s >= score_thr and x >= 0 and y >= 0:
            px, py = int(x), int(y)
            cv2.circle(img, (px, py), 4, (0, 255, 0), -1, cv2.LINE_AA)
            label = str(idx + 1)
            # black shadow for readability
            cv2.putText(img, label, (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            # white foreground
            cv2.putText(img, label, (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    measurements = compute_measurements_for_item(kpts_flat, category_id, unit, px_per_cm, score_thr=score_thr)
    unit_lbl = measurements.get('_unit', unit)
    # Determine group for drawing logic
    group_key = _category_group(category_id)

    # Draw measurement lines using keypoints
    colors = {
        # generic
        'body_length': (0, 255, 0),
        # 'outseam_length': (0, 255, 0),
        # tops KP-based
        'neck': (255, 140, 0),   # dark orange
        'sleeve_length': (255, 0, 0),        # blue-like
        'shoulder_to_shoulder': (0, 165, 255),
        'chest': (255, 0, 0),
        'waist': (0, 0, 255),
        'hem': (0, 255, 255),
        'sleeve': (128, 0, 128),
        'front_length': (0, 255, 0),
        # bottoms (kept for completeness)
        'inseam': (255, 0, 255),
        'lenght': (255, 0, 165),
        'half_knee': (255, 0, 0),
        'leg_opening': (0, 165, 255),
        'front_rise': (0, 255, 0),
        'back_rise': (0, 0, 128),
    }

    if group_key in ('G1_SHORT_TOP_OUTWEAR', 'G2_LONG_TOP_OUTWEAR', 'G3_VEST', 'G10_SHORT_DRESS', 'G11_LONG_DRESS'):
        # Draw only the segments defined for this group
        for name in [seg['name'] for seg in KP_SEGMENTS_BY_GROUP.get(group_key, [])]:
            line = measurements.get(f'{name}_line')
            if line and name in measurements:
                pL = (line['left'][0], line['left'][1])
                pR = (line['right'][0], line['right'][1])
                _draw_line_with_label(img, pL, pR, f"{name.replace('_',' ')}: {measurements[name]:.1f} {unit_lbl}", colors.get(name, (255,255,255)))
        # no generic body length drawing for tops/dresses; use front length only
        if 'front_length_line' in measurements:
            top = measurements['front_length_line']['left']
            bot = measurements['front_length_line']['right']
            _draw_line_with_label(img, (top[0], top[1]), (bot[0], bot[1]), f"front length: {measurements['front_length']:.1f} {unit_lbl}", colors['body_length'])

    if group_key in ('G4_SHORTS', 'G5_TROUSERS', 'G9_SKIRT'):
        # Draw KP-based segments first if any
        for seg in KP_SEGMENTS_BY_GROUP.get(group_key, []):
            name = seg['name']
            line = measurements.get(f'{name}_line')
            if line and name in measurements:
                pL = (line['left'][0], line['left'][1])
                pR = (line['right'][0], line['right'][1])
                _draw_line_with_label(img, pL, pR, f"{name.replace('_',' ')}: {measurements[name]:.1f} {unit_lbl}", colors.get(name, (255,255,255)))
        # Also draw band-based widths if they exist
        for name in ['waist_width', 'hip_width', 'thigh_width', 'hem_width']:
            line = measurements.get(f'{name}_line')
            if line and name in measurements:
                pL = (line['left'][0], line['left'][1])
                pR = (line['right'][0], line['right'][1])
                _draw_line_with_label(img, pL, pR, f"{name.replace('_',' ')}: {measurements[name]:.1f} {unit_lbl}", colors.get(name, (255,255,255)))
        # vertical length
        # Hide outseam length for bottoms

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
                     limit: Optional[int] = None,
                     only_image_id: Optional[int] = None,
                     only_file: Optional[str] = None) -> None:
    with open(results_json, 'r') as f:
        results = json.load(f)

    coco = _load_coco(ann_json)
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
        if only_image_id is not None and int(img_id) != int(only_image_id):
            continue
        if only_file is not None and str(fname) != str(only_file):
            continue
        image_path = os.path.join(images_dir, fname)
        out_path = os.path.join(out_vis_dir, fname)
        try:
            measurements = visualize_item(image_path, kpts, category_id, unit, px_per_cm, out_path, score_thr=score_thr)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
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

    parser = argparse.ArgumentParser(description='DeepFashion2 measurement and visualization from keypoints')
    parser.add_argument('--results', required=True, help='Path to keypoints results JSON (from tools/test.py)')
    parser.add_argument('--ann', required=True, help='Path to annotation JSON (val-coco_style.json)')
    parser.add_argument('--images-dir', required=True, help='Path to validation images directory')
    parser.add_argument('--out-vis-dir', default='output/measure_vis_keypoints', help='Directory to save visualizations')
    parser.add_argument('--out-report', default='output/measure_report_keypoints.json', help='Output JSON with measurements')
    parser.add_argument('--unit', default='cm', choices=['cm', 'inch', 'px'])
    parser.add_argument('--px-per-cm', type=float, default=None, help='Pixels per centimeter (required for cm/inch)')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Keypoint score threshold for visibility')
    parser.add_argument('--limit', type=int, default=None, help='Optional, limit number of items to process')
    parser.add_argument('--only-image-id', type=int, default=None, help='Process only this image_id (e.g., 63 for 000063.jpg)')
    parser.add_argument('--only-file', type=str, default=None, help='Process only this file name (e.g., 000063.jpg)')

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
        only_image_id=args.only_image_id,
        only_file=args.only_file,
    )
