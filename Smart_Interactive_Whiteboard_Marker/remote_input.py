"""Presentation remote = a USB HID keyboard. Its two buttons send ordinary key events
(usually Page Down / Page Up, sometimes Right / Left arrows or F5 / Esc).
We read them with cv2.waitKeyEx(), which returns the full key code (waitKey() truncates it).
The application window must have the keyboard focus.

Mapping:  NEXT button -> LEFT click     PREVIOUS button -> RIGHT click   (changeable in config.json)
Learning mode records the real codes of the remote (and logs every code to remote_keys.log).
"""
import time

LOG_PATH = "remote_keys.log"


class RemoteInput:
    DEBOUNCE = 0.35          # s: ignore key auto-repeat while a button is held

    def __init__(self, cfg):
        self.left = set(cfg["remote"]["left"])
        self.right = set(cfg["remote"]["right"])
        self.last_code = None
        self._last_fire = 0.0
        self.learning = False
        self.done = False
        self.step = 0
        self.collected = [[], []]
        self._t0 = None

    def to_config(self):
        return {"left": sorted(self.left), "right": sorted(self.right)}

    # ---- learning ----------------------------------------------------------------------
    def start_learning(self):
        self.learning, self.done, self.step = True, False, 0
        self.collected, self._t0 = [[], []], None

    def cancel_learning(self):
        self.learning = False

    def status(self):
        if self.step == 0:
            return "Press the PREVIOUS button on the remote (right click)"
        return "Now press the NEXT button on the remote (left click)"

    # ---- events ------------------------------------------------------------------------
    def poll(self, key, now=None):
        """key = value of cv2.waitKeyEx (-1 if none). Returns 'left', 'right' or None."""
        now = time.time() if now is None else now
        if self.learning and self._t0 is not None and now - self._t0 > 0.6:   # button finished
            self._next_step()
        if key == -1:
            return None
        self.last_code = key
        print("key code:", key)
        with open(LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {key}\n")
        if self.learning:
            if key not in self.collected[self.step]:
                self.collected[self.step].append(key)      # one press may send several codes
            if self._t0 is None:
                self._t0 = now
            return None
        if now - self._last_fire < self.DEBOUNCE:
            return None
        for name, codes in (("left", self.left), ("right", self.right)):
            if key in codes:
                self._last_fire = now
                return name
        return None

    def _next_step(self):
        self.step += 1
        self._t0 = None
        if self.step == 2:
            self.right, self.left = set(self.collected[0]), set(self.collected[1])
            self.learning, self.done = False, True
