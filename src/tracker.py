import cv2
import time
import math

class Track:
    def __init__(self, tid, bbox, frame, tracker_type='CSRT'):
        self.id = tid
        if tracker_type == 'CSRT':
            self.tracker = cv2.TrackerCSRT_create()
        else:
            self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, tuple(bbox))
        x,y,w,h = bbox
        self.centroids = [((x+w/2),(y+h/2))]
        self.last_seen = time.time()

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        if not ok:
            return False, None
        x,y,w,h = [int(v) for v in box]
        cx = x + w/2
        cy = y + h/2
        self.centroids.append((cx,cy))
        self.last_seen = time.time()
        return True, (x,y,w,h)

class TrackManager:
    def __init__(self, max_age=2.0):
        self.tracks = {}
        self._next_id = 1
        self.max_age = max_age

    def add_detections(self, boxes, frame):
        # naive: create a new track for each box
        for box in boxes:
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(tid, box, frame)

    def update(self, frame):
        alive = {}
        for tid, tr in list(self.tracks.items()):
            ok, box = tr.update(frame)
            if ok:
                alive[tid] = (tr, box)
        # remove stale tracks
        now = time.time()
        for tid in list(self.tracks.keys()):
            if tid not in alive:
                tr = self.tracks[tid]
                if now - tr.last_seen > self.max_age:
                    del self.tracks[tid]
        return alive

def compute_speed(track, window=10):
    pts = track.centroids[-window:]
    if len(pts) < 2:
        return 0.0
    s = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        s += math.hypot(dx,dy)
    return s / (len(pts)-1)
