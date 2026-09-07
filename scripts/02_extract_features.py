"""
Extract per-frame proximity/motion features from the pattern-cutting video
using the YOLO11n-seg tip-tracking model, and attach collision / near-miss
labels from data/ground_truth.csv.

For every sampled frame we run the segmentation model and, per class
(TIPL, TIPR, TIPandCircle), keep the highest-confidence detection. From that
we derive:
  - 2D centroid position of each tip/marker
  - pairwise distance between TIPL and TIPR (the main collision proxy - two
    instrument tips colliding means this distance drops toward 0)
  - bounding-box IoU between TIPL and TIPR (direct overlap proxy)
  - segmentation mask area of each detection, used as a monocular
    "pseudo-depth" cue - an object appears larger the closer it is to the
    camera, so a sudden area change often coincides with the arms moving
    toward/away from the camera during a collision
  - frame-to-frame velocity of each tip (speed of approach)
  - a constant-velocity tracked position for TIPL/TIPR that keeps
    extrapolating through brief detection gaps (see SimpleTracker) - the
    detector loses track of a tip 56% of the time during an actual labeled
    collision (both tips visible together only 43.5% of the time on
    collision frames vs 79% elsewhere) because the tips occlude each other
    exactly when they touch, so the raw distance/IoU features go blind at
    the moment that matters most. The tracked distance stays meaningful
    through that occlusion instead of falling back to a sentinel.

Each row is labeled using the time windows in data/ground_truth.csv
(padded by --label-window seconds to absorb annotation imprecision).

Usage:
    python scripts/02_extract_features.py --sample-fps 5
    python scripts/02_extract_features.py --sample-fps 5 --max-frames 500   # quick test
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "video.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "video_tip_circle_yolo11n_seg_best.pt"
GROUND_TRUTH_CSV = PROJECT_ROOT / "data" / "ground_truth.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "features_and_labels.csv"

# Known classes baked into models/video_tip_circle_yolo11n_seg_best.pt
TIP_L, TIP_R, CIRCLE = "TIPL", "TIPR", "TIPandCircle"


class SimpleTracker:
    """Constant-velocity tracker for one point. While the detector sees the
    object, it just records position/velocity. While the object is
    undetected (occluded), it keeps extrapolating position from the last
    known velocity, up to max_occlusion_s - long enough to bridge a brief
    collision, short enough that we stop trusting a guess that's gone stale."""

    def __init__(self, max_occlusion_s: float):
        self.x = self.y = self.vx = self.vy = 0.0
        self.last_seen_t = None
        self.max_occlusion_s = max_occlusion_s

    def update(self, t: float, x: float | None, y: float | None):
        """Returns (est_x, est_y, state) where state is 'detected',
        'predicted', or 'lost' (est_x/est_y are None when lost)."""
        if x is not None:
            if self.last_seen_t is not None:
                dt = max(t - self.last_seen_t, 1e-6)
                self.vx = (x - self.x) / dt
                self.vy = (y - self.y) / dt
            self.x, self.y, self.last_seen_t = x, y, t
            return self.x, self.y, "detected"

        if self.last_seen_t is None:
            return None, None, "lost"
        dt = t - self.last_seen_t
        if dt > self.max_occlusion_s:
            return None, None, "lost"
        return self.x + self.vx * dt, self.y + self.vy * dt, "predicted"


def load_ground_truth(path):
    events = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            events.append(
                {
                    "start": float(row["start_time_s"]),
                    "end": float(row["end_time_s"]),
                    "is_collision": int(row["is_collision"]),
                    "near_miss": int(row["near_miss"]),
                    "severity": int(row["collision_severity"]) if row["collision_severity"] else 0,
                    "matched": False,
                }
            )
    return events


def label_for_time(t, events, window_s):
    """Label a timestamp using any ground-truth event whose window (padded
    by window_s to account for annotation imprecision) covers it. Marks each
    matched event so main() can flag any that never lined up with a sampled
    frame (a sign of a timestamp/frame-rate misalignment bug)."""
    label, near_miss, severity = 0, 0, 0
    for ev in events:
        if ev["start"] - window_s <= t <= ev["end"] + window_s:
            ev["matched"] = True
            if ev["is_collision"]:
                label = 1
                severity = max(severity, ev["severity"])
            if ev["near_miss"]:
                near_miss = 1
    return label, near_miss, severity


def check_label_alignment(events):
    unmatched = [ev for ev in events if not ev["matched"]]
    if unmatched:
        print(f"WARNING: {len(unmatched)}/{len(events)} ground-truth events never matched a "
              f"sampled frame (check sample-fps isn't too sparse, or a timestamp typo):")
        for ev in unmatched[:10]:
            print(f"    start={ev['start']}s end={ev['end']}s collision={ev['is_collision']} "
                  f"near_miss={ev['near_miss']}")
    else:
        print(f"Label alignment OK: all {len(events)} ground-truth events matched >=1 sampled frame.")


def best_detection_per_class(result, names):
    """Return {class_name: (cx, cy, w, h, conf, mask_area)} keeping only the
    highest-confidence box per class."""
    best = {}
    if result.boxes is None or len(result.boxes) == 0:
        return best
    boxes = result.boxes
    masks = result.masks
    for i in range(len(boxes)):
        cls_name = names[int(boxes.cls[i])]
        conf = float(boxes.conf[i])
        if cls_name in best and best[cls_name][4] >= conf:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        if masks is not None:
            mask_arr = masks.data[i].cpu().numpy() > 0.5
            mask_area = float(mask_arr.sum())
        else:
            mask_arr = None
            mask_area = w * h
        best[cls_name] = (cx, cy, w, h, conf, mask_area, mask_arr)
    return best


def mask_overlap_features(mask_by_name, a, b):
    """True per-pixel segmentation-mask overlap between two detections, as
    opposed to bounding-box IoU which counts two tips as 'overlapping'
    whenever their boxes cross even if the tips themselves don't touch.
    Returns (mask_iou, intersection_area, overlap_frac_of_smaller)."""
    ma, mb = mask_by_name.get(a), mask_by_name.get(b)
    if ma is None or mb is None:
        return 0.0, 0.0, 0.0
    inter = float(np.logical_and(ma, mb).sum())
    area_a, area_b = float(ma.sum()), float(mb.sum())
    union = area_a + area_b - inter
    mask_iou = inter / union if union > 0 else 0.0
    smaller = min(area_a, area_b)
    overlap_frac = inter / smaller if smaller > 0 else 0.0
    return mask_iou, inter, overlap_frac


def dist(row, a, b):
    if row[f"{a}_present"] and row[f"{b}_present"]:
        return float(np.hypot(row[f"{a}_x"] - row[f"{b}_x"], row[f"{a}_y"] - row[f"{b}_y"]))
    return -1.0  # sentinel: at least one tip not visible this frame


def iou(row, a, b):
    if not (row[f"{a}_present"] and row[f"{b}_present"]):
        return 0.0
    ax1, ay1 = row[f"{a}_x"] - row[f"{a}_w"] / 2, row[f"{a}_y"] - row[f"{a}_h"] / 2
    ax2, ay2 = row[f"{a}_x"] + row[f"{a}_w"] / 2, row[f"{a}_y"] + row[f"{a}_h"] / 2
    bx1, by1 = row[f"{b}_x"] - row[f"{b}_w"] / 2, row[f"{b}_y"] - row[f"{b}_h"] / 2
    bx2, by2 = row[f"{b}_x"] + row[f"{b}_w"] / 2, row[f"{b}_y"] + row[f"{b}_h"] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = row[f"{a}_w"] * row[f"{a}_h"] + row[f"{b}_w"] * row[f"{b}_h"] - inter
    return inter / union if union > 0 else 0.0


def add_temporal_features(df: pd.DataFrame, sample_fps: float) -> pd.DataFrame:
    """Add rolling/derivative features that capture collision *dynamics*
    (closing speed, sustained proximity) instead of a single noisy frame.
    A single frame's raw distance is easily thrown off by a missed detection
    or jitter; these features smooth over that and give the model a sense of
    trajectory, which single-frame features can't."""
    window = max(1, round(sample_fps * 1.0))  # ~1 second of frames

    # Prefer the tracker's occlusion-bridged distance where available (it
    # stays valid through brief occlusion instead of dropping out exactly
    # when tips are colliding); fall back to the raw sentinel-based series
    # for the small remainder still interpolated.
    dist = df["dist_TIPL_TIPR_tracked"].replace(-1.0, np.nan)
    dist = dist.fillna(df["dist_TIPL_TIPR"].replace(-1.0, np.nan)).interpolate(limit=window)
    smooth = dist.rolling(3, min_periods=1, center=True).mean()

    df["dist_TIPL_TIPR_smooth"] = smooth.fillna(-1.0)
    df["dist_TIPL_TIPR_delta"] = smooth.diff().fillna(0.0)  # negative = closing in
    df["dist_TIPL_TIPR_rollmin_1s"] = dist.rolling(window, min_periods=1).min().fillna(-1.0)
    df["iou_TIPL_TIPR_rollmax_1s"] = df["iou_TIPL_TIPR"].rolling(window, min_periods=1).max()
    df["mask_iou_TIPL_TIPR_rollmax_1s"] = df["mask_iou_TIPL_TIPR"].rolling(window, min_periods=1).max()
    df["mask_overlap_frac_TIPL_TIPR_rollmax_1s"] = df["mask_overlap_frac_TIPL_TIPR"].rolling(window, min_periods=1).max()
    df["combined_tip_speed"] = df["TIPL_speed"] + df["TIPR_speed"]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(VIDEO_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--sample-fps", type=float, default=5.0,
                         help="Frames per second to sample & run inference on")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--label-window", type=float, default=0.5,
                         help="Seconds to pad each ground-truth event window by")
    parser.add_argument("--max-occlusion-s", type=float, default=1.5,
                         help="How long to keep extrapolating a tip's position through occlusion")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Stop after this many sampled frames (for quick tests)")
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    args = parser.parse_args()

    events = load_ground_truth(GROUND_TRUTH_CSV)
    model = YOLO(args.model)
    names = model.names

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_step = max(1, round(native_fps / args.sample_fps))
    print(f"Video: {native_fps:.2f} fps, {total_frames:.0f} frames -> "
          f"sampling every {frame_step} frames (~{args.sample_fps} fps)")

    rows = []
    prev = {}  # class_name -> (cx, cy, t)
    frame_idx = 0
    sampled = 0
    classes = [TIP_L, TIP_R, CIRCLE]
    prev_both_present = None
    last_both_time = None
    tracker_L = SimpleTracker(args.max_occlusion_s)
    tracker_R = SimpleTracker(args.max_occlusion_s)

    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = frame_idx / native_fps

        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        det = best_detection_per_class(result, names)

        row = {"frame_idx": frame_idx, "time_s": round(t, 3)}
        mask_by_name = {}
        for name in classes:
            if name in det:
                cx, cy, w, h, conf, area, mask_arr = det[name]
                if mask_arr is not None:
                    mask_by_name[name] = mask_arr
            else:
                cx = cy = w = h = conf = area = 0.0
            row[f"{name}_present"] = 1 if name in det else 0
            row[f"{name}_x"] = cx
            row[f"{name}_y"] = cy
            row[f"{name}_w"] = w
            row[f"{name}_h"] = h
            row[f"{name}_conf"] = conf
            row[f"{name}_area"] = area

        row["dist_TIPL_TIPR"] = dist(row, TIP_L, TIP_R)
        row["dist_TIPL_CIRCLE"] = dist(row, TIP_L, CIRCLE)
        row["dist_TIPR_CIRCLE"] = dist(row, TIP_R, CIRCLE)
        row["iou_TIPL_TIPR"] = iou(row, TIP_L, TIP_R)

        mask_iou, mask_inter, mask_overlap_frac = mask_overlap_features(mask_by_name, TIP_L, TIP_R)
        row["mask_iou_TIPL_TIPR"] = mask_iou
        row["mask_overlap_area_TIPL_TIPR"] = mask_inter
        row["mask_overlap_frac_TIPL_TIPR"] = mask_overlap_frac

        # A tip vanishing from detection is often occlusion from the other
        # tip covering it during an actual collision, not a detector fluke -
        # both-present rate on labeled collision frames is ~44% vs ~79%
        # elsewhere, so "did the pair just drop out" is itself a signal.
        both_present_now = bool(row[f"{TIP_L}_present"] and row[f"{TIP_R}_present"])
        row["tip_pair_dropped_this_frame"] = 1 if (prev_both_present and not both_present_now) else 0
        if both_present_now:
            last_both_time = t
        row["time_since_both_present"] = (t - last_both_time) if last_both_time is not None else -1.0
        prev_both_present = both_present_now

        lx, ly, lstate = tracker_L.update(t, row[f"{TIP_L}_x"] if row[f"{TIP_L}_present"] else None,
                                           row[f"{TIP_L}_y"] if row[f"{TIP_L}_present"] else None)
        rx, ry, rstate = tracker_R.update(t, row[f"{TIP_R}_x"] if row[f"{TIP_R}_present"] else None,
                                           row[f"{TIP_R}_y"] if row[f"{TIP_R}_present"] else None)
        track_state_code = {"detected": 0, "predicted": 1, "lost": 2}
        row["TIPL_track_state"] = track_state_code[lstate]
        row["TIPR_track_state"] = track_state_code[rstate]
        row["dist_TIPL_TIPR_tracked"] = (
            float(np.hypot(lx - rx, ly - ry)) if lx is not None and rx is not None else -1.0
        )

        for name in classes:
            if row[f"{name}_present"] and name in prev:
                pcx, pcy, pt = prev[name]
                dt = max(t - pt, 1e-6)
                row[f"{name}_speed"] = float(np.hypot(row[f"{name}_x"] - pcx, row[f"{name}_y"] - pcy) / dt)
            else:
                row[f"{name}_speed"] = 0.0
            if row[f"{name}_present"]:
                prev[name] = (row[f"{name}_x"], row[f"{name}_y"], t)

        label, near_miss, severity = label_for_time(t, events, args.label_window)
        row["label_collision"] = label
        row["label_near_miss"] = near_miss
        row["label_severity"] = severity

        rows.append(row)
        sampled += 1
        frame_idx += 1

        if sampled % 200 == 0:
            print(f"  ...processed {sampled} sampled frames (t={t:.1f}s)")
        if args.max_frames is not None and sampled >= args.max_frames:
            break

    cap.release()
    check_label_alignment(events)

    if not rows:
        raise RuntimeError("No frames were processed - check --video / --sample-fps")

    # cap.grab() returning False mid-video (e.g. an FFmpeg stream timeout
    # from disk/network contention) looks identical to reaching the real end
    # of the video, so a truncated run would otherwise exit silently with a
    # seemingly-valid but incomplete CSV.
    expected_frames = total_frames if args.max_frames is None else None
    if expected_frames and frame_idx < expected_frames - frame_step:
        raise RuntimeError(
            f"Video reading stopped early at frame {frame_idx}/{expected_frames:.0f} "
            f"(only {sampled} of an expected ~{expected_frames / frame_step:.0f} sampled frames). "
            "This usually means the video decoder hit an I/O error/timeout, not the real end "
            "of the video (e.g. another process reading the same file, or a cloud-sync stall). "
            "Close other programs touching data/video.mp4 and re-run."
        )

    df = pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)
    df = add_temporal_features(df, args.sample_fps)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    n_pos = int(df["label_collision"].sum())
    print(f"Wrote {len(df)} rows to {args.output}  "
          f"(positive frames: {n_pos}, {100 * n_pos / len(df):.1f}%)")


if __name__ == "__main__":
    main()
