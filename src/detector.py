import cv2
import numpy as np

class MotionDetector:
    def __init__(self, min_area=5000):
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.min_area = min_area

    def detect(self, frame):
        """Return list of bboxes (x,y,w,h) for moving regions in the frame."""
        fg = self.backSub.apply(frame)
        # morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.dilate(fg, kernel, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = frame.shape[:2]
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x,y,ww,hh = cv2.boundingRect(c)
            # clamp
            x = max(0, x); y = max(0, y)
            ww = min(w-x, ww); hh = min(h-y, hh)
            boxes.append((x,y,ww,hh))
        return boxes
