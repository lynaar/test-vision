"""Free-drawing layer: pen/eraser rendered with our own primitives (disc stamping along a segment).
The layer keeps a colour image + a boolean mask so it can be composed over any background
(white sheet for free drawing, lesson page for annotations)."""
import numpy as np

PALETTE = [("Black", (40, 40, 40)), ("Blue", (190, 90, 20)), ("Red", (40, 40, 220)), ("Green", (60, 150, 40))]
THICKNESSES = [3, 7, 14]          # pen diameters in pixels
_disc_cache = {}


def _disc(r):
    if r not in _disc_cache:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        _disc_cache[r] = (xx * xx + yy * yy) <= r * r
    return _disc_cache[r]


class DrawingLayer:
    MAX_UNDO = 12

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.color = np.zeros((h, w, 3), np.uint8)
        self.mask = np.zeros((h, w), bool)
        self.pen_color = PALETTE[0][1]
        self.thickness = THICKNESSES[1]
        self.eraser = False
        self.pen_down = False
        self._last = None
        self._undo = []

    @property
    def radius(self):
        return (self.thickness * 2 + 4) if self.eraser else max(1, self.thickness // 2)

    def set_color(self, c):
        self.pen_color, self.eraser = c, False

    def set_thickness(self, t):
        self.thickness = t

    def toggle_eraser(self):
        self.eraser = not self.eraser

    # ---- strokes -----------------------------------------------------------------------
    def begin_stroke(self):
        self._undo.append((self.color.copy(), self.mask.copy()))
        self._undo = self._undo[-self.MAX_UNDO:]
        self.pen_down, self._last = True, None

    def end_stroke(self):
        self.pen_down, self._last = False, None

    def _stamp(self, x, y):
        r = self.radius
        x0, y0 = x - r, y - r
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(self.w, x + r + 1), min(self.h, y + r + 1)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        d = _disc(r)[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        if self.eraser:
            self.mask[cy0:cy1, cx0:cx1][d] = False
        else:
            self.mask[cy0:cy1, cx0:cx1][d] = True
            self.color[cy0:cy1, cx0:cx1][d] = self.pen_color

    def move(self, pt):
        """Pen moves to pt: stamp discs along the segment from the previous point (no gaps)."""
        if not self.pen_down:
            return
        x, y = int(pt[0]), int(pt[1])
        if self._last is None:
            self._stamp(x, y)
        else:
            lx, ly = self._last
            steps = int(np.hypot(x - lx, y - ly) / max(1.0, self.radius / 2)) + 1
            for i in range(1, steps + 1):
                t = i / steps
                self._stamp(int(round(lx + t * (x - lx))), int(round(ly + t * (y - ly))))
        self._last = (x, y)

    def undo(self):
        self.end_stroke()
        if self._undo:
            self.color, self.mask = self._undo.pop()

    def clear(self):
        self.end_stroke()
        self._undo.append((self.color.copy(), self.mask.copy()))
        self.color[:] = 0
        self.mask[:] = False

    def compose(self, base):
        base[self.mask] = self.color[self.mask]
        return base
