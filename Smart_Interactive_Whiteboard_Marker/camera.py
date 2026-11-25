"""Camera acquisition. A background thread always keeps the LATEST frame so that slow
processing never builds up a delay (the pointer would lag behind the real marker)."""
import threading
import cv2


class Camera:
    def __init__(self, index=0, size=(1280, 720)):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.ok = self.cap.isOpened()
        self._frame = None
        self._lock = threading.Lock()
        self._running = self.ok
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            ok, f = self.cap.read()          # no flip: board coordinates must not be mirrored
            if ok:
                with self._lock:
                    self._frame = f

    def read(self):
        """Latest frame (BGR) or None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        self.cap.release()
