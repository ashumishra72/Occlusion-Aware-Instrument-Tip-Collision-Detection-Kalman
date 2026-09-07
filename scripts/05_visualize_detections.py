"""
Play the video in a live OpenCV window with YOLO tip/marker detections drawn
on top, so you can visually check what the model is actually seeing
(useful for sanity-checking detection quality, e.g. the TIPL dropout issue
discussed while building the pipeline).

Controls while the window is focused:
    space   - pause / resume
    q / Esc - quit
    n       - step one sampled frame forward while paused

Usage:
    python scripts/05_visualize_detections.py
    python scripts/05_visualize_detections.py --start-sec 50           # jump to a timestamp
    python scripts/05_visualize_detections.py --sample-fps 10 --conf 0.1
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "video.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "video_tip_circle_yolo11n_seg_best.pt"

COLORS = {"TIPL": (0, 255, 0), "TIPR": (255, 100, 0), "TIPandCircle": (0, 215, 255)}


def draw_detections(frame, result, names):
    if result.boxes is not None:
        for i in range(len(result.boxes)):
            cls_name = names[int(result.boxes.cls[i])]
            conf = float(result.boxes.conf[i])
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].tolist())
            color = COLORS.get(cls_name, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if result.masks is not None:
        overlay = frame.copy()
        for i in range(len(result.masks.data)):
            cls_name = names[int(result.boxes.cls[i])]
            color = COLORS.get(cls_name, (200, 200, 200))
            mask = result.masks.data[i].cpu().numpy()
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            overlay[mask > 0.5] = color
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, dst=frame)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(VIDEO_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--sample-fps", type=float, default=5.0,
                         help="How many frames per second to run detection on and display")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--start-sec", type=float, default=0.0,
                         help="Jump to this timestamp before playing")
    parser.add_argument("--display-scale", type=float, default=0.6,
                         help="Resize factor for the display window (video is 1920x1080)")
    args = parser.parse_args()

    model = YOLO(args.model)
    names = model.names

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, round(native_fps / args.sample_fps))

    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_sec * native_fps))

    print("Controls: space = pause/resume, n = step forward (while paused), q/Esc = quit")

    paused = False
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while True:
        if not paused:
            ok = cap.grab()
            if not ok:
                print("End of video.")
                break
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = frame_idx / native_fps

            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            frame = draw_detections(frame, result, names)
            cv2.putText(frame, f"t={t:.2f}s  frame={frame_idx}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if args.display_scale != 1.0:
                frame = cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale)
            cv2.imshow("Tip detection (space=pause, q=quit)", frame)
            frame_idx += 1

        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("n") and paused:
            paused = False  # advance one sampled frame then re-pause on next loop
            cv2.waitKey(1)
            paused = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
