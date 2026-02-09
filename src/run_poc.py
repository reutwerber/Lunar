import yaml
import time
import json
import cv2
import argparse
from camera import open_capture
from detector import MotionDetector
from tracker import TrackManager, compute_speed

def overlay(frame, tracks):
    for tid, (tr, box) in tracks.items():
        x,y,w,h = [int(v) for v in box]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        speed = compute_speed(tr)
        label = f"ID:{tid} S:{speed:.1f}"
        cv2.putText(frame, label, (x, max(10,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

def main(cfg):
    cap = open_capture(cfg.get('source', 0))
    if not cap or not cap.isOpened():
        print(json.dumps({"error":"cannot open source"}))
        return

    detect_interval = int(cfg.get('detect_interval', 30))
    motion_min_area = cfg.get('motion_min_area', 5000)
    nervous_window = int(cfg.get('nervous_window', 30))
    nervous_speed_thresh = float(cfg.get('nervous_speed_thresh', 5.0))
    show = bool(cfg.get('show_window', True))

    detector = MotionDetector(min_area=motion_min_area)
    tracks = TrackManager()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        frame_idx += 1

        if frame_idx % detect_interval == 0:
            boxes = detector.detect(frame)
            if boxes:
                tracks.add_detections(boxes, frame)

        alive = tracks.update(frame)

        # compute nervousness per track
        events = []
        for tid, (tr, box) in alive.items():
            speed = compute_speed(tr, window=nervous_window)
            nervous = speed > nervous_speed_thresh
            x,y,w,h = [int(v) for v in box]
            events.append({
                "id": tid,
                "bbox": [x,y,w,h],
                "speed": speed,
                "nervous": bool(nervous)
            })

        if events:
            out = {"timestamp": time.time(), "hasDog": True, "tracks": events}
            print(json.dumps(out))

        if show:
            overlay(frame, alive)
            cv2.imshow('poc', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/camera.yml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
