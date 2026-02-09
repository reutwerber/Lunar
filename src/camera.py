import cv2

def open_capture(source):
    """Open a cv2.VideoCapture for the given source.

    source: int index or string URL
    """
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
        return cap
    except Exception:
        # treat as URL
        cap = cv2.VideoCapture(source)
        return cap
