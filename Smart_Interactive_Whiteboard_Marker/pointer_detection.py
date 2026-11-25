"""Pointer (colour marker) detection, written with numpy.

Method: colour-marker segmentation in HSV, because hue is mostly independent of brightness, and a
saturated marker is rare on a white board.  Pipeline:
  1. sub-sample the frame x2 (speed);   2. BGR -> HSV (own implementation);
  3. threshold on hue distance / saturation / value;  4. keep only the board region (ROI);
  5. remove pixels that the PROJECTOR itself paints in the marker colour (exclusion mask);
  6. morphological opening then closing (own box-filter erosion/dilation) to remove noise;
  7. connected components (cv2) + shape filter (area, fill ratio, aspect) + own centroid;
  8. tracking: nearest candidate to the previous one, N confirmation frames, loss tolerance,
     adaptive exponential smoothing (steady when slow, reactive when fast).
Limits: marker must stay visible and not be covered by the hand; same-coloured projected content is
masked (the marker cannot be seen above it); strong ambient light changes need a re-calibration.
"""
import numpy as np
import cv2


# ------------------------------------------------------------------------- colour conversion
def bgr_to_hsv(img):
    """Own BGR->HSV (OpenCV convention: H 0..179, S and V 0..255). Returns (h, s, v) int arrays."""
    b, g, r = [img[..., i].astype(np.float32) for i in range(3)]
    v = np.maximum(np.maximum(b, g), r)
    d = v - np.minimum(np.minimum(b, g), r)
    s = np.where(v > 0, d * 255.0 / np.maximum(v, 1), 0)
    dd = np.maximum(d, 1e-6)
    sector = np.where(v == r, ((g - b) / dd) % 6, np.where(v == g, (b - r) / dd + 2, (r - g) / dd + 4))
    h = np.where(d == 0, 0, sector * 30)         # 60 degrees / 2 = 30 per sector unit
    return h.astype(np.int16) % 180, s.astype(np.int16), v.astype(np.int16)


def hue_distance(h, h0):
    """Circular distance between hues (0 and 179 are neighbours)."""
    return np.abs(((h.astype(np.int16) - int(h0) + 90) % 180) - 90)


# ------------------------------------------------------------------------- morphology
def box_sum(m, k):
    """Sum of a k x k window around each pixel, computed with an integral image."""
    r = k // 2
    p = np.pad(m.astype(np.int32), r)
    ii = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]


def erode(m, k):
    return box_sum(m, k) == k * k


def dilate(m, k):
    return box_sum(m, k) > 0


# ------------------------------------------------------------------------- tracker
class PointerTracker:
    DS = 2                 # sub-sampling factor
    MIN_AREA = 10          # in sub-sampled pixels
    MAX_AREA_RATIO = 0.01
    MIN_FILL = 0.45        # blob area / bounding-box area (a marker is compact)
    MAX_ASPECT = 4.0
    CONFIRM = 3            # consecutive frames before the pointer is reported
    LOST = 8               # frames without detection tolerated (last position is kept)
    RADIUS = 60            # tracking gate in sub-sampled pixels

    def __init__(self, hsv_ref=None):
        self.hsv_ref = tuple(hsv_ref) if hsv_ref else None
        self.reset()

    def reset(self):
        self.prev = None
        self.smooth = None
        self.hits = 0
        self.miss = 0

    def ref_hue(self):
        return 0 if self.hsv_ref is None else self.hsv_ref[0]

    def calibrate(self, frame, pt, radius=5):
        """Learn the marker colour from a patch around the clicked pixel."""
        x, y = int(pt[0]), int(pt[1])
        patch = frame[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
        if patch.size == 0:
            return False
        h, s, v = bgr_to_hsv(patch)
        ang = h.astype(float) * 2 * np.pi / 180            # circular mean of the hue
        hue = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 180 / (2 * np.pi)) % 180
        self.hsv_ref = (int(hue), int(np.median(s)), int(np.median(v)))
        self.reset()
        return True

    def forget_color(self):
        self.hsv_ref = None
        self.reset()

    def _color_mask(self, hsv):
        h, s, v = hsv
        if self.hsv_ref is None:                            # default: vivid red
            return (hue_distance(h, 0) <= 8) & (s >= 150) & (v >= 110)
        h0, s0, v0 = self.hsv_ref
        return (hue_distance(h, h0) <= 10) & (s >= max(80, s0 - 80)) & (v >= max(60, v0 - 100))

    def detect(self, frame, roi=None, exclude=None):
        """Returns (debug mask, (x, y) in full-resolution camera pixels or None)."""
        ds = self.DS
        m = self._color_mask(bgr_to_hsv(frame[::ds, ::ds]))
        if roi is not None:
            m &= roi[::ds, ::ds] > 0
        if exclude is not None:
            m &= ~(exclude[::ds, ::ds] > 0)
        m = dilate(erode(m, 3), 3)           # opening: removes isolated noise pixels
        m = erode(dilate(m, 5), 5)           # closing: fills small holes in the marker
        mask8 = m.astype(np.uint8) * 255
        h, w = m.shape
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask8, connectivity=8)

        cands = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < self.MIN_AREA or area > self.MAX_AREA_RATIO * w * h:
                continue
            if area / float(bw * bh) < self.MIN_FILL or max(bw, bh) / max(1, min(bw, bh)) > self.MAX_ASPECT:
                continue
            ys, xs = np.nonzero(labels[y:y + bh, x:x + bw] == i)
            cands.append((area, (x + xs.mean(), y + ys.mean())))     # centroid = mean of pixel coords

        if not cands:
            self.miss += 1
            if self.miss > self.LOST:
                self.reset()
            return mask8, self._output()

        dist = lambda c: np.hypot(c[1][0] - self.prev[0], c[1][1] - self.prev[1])
        near = [c for c in cands if self.prev is not None and dist(c) < self.RADIUS]
        if near:
            best = min(near, key=dist)
            self.hits += 1
        else:                                  # new object: restart confirmation
            best = max(cands, key=lambda c: c[0])
            self.hits, self.smooth = 1, None
        self.miss = 0
        self.prev = best[1]
        full = np.array([best[1][0] * ds + ds / 2, best[1][1] * ds + ds / 2])
        if self.smooth is None:
            self.smooth = full
        else:
            speed = np.hypot(*(full - self.smooth))
            a = min(0.9, 0.3 + speed / 80.0)   # fast motion -> follow, slow -> filter jitter
            self.smooth = a * full + (1 - a) * self.smooth
        return mask8, self._output()

    def _output(self):
        if self.hits >= self.CONFIRM and self.smooth is not None:
            return (float(self.smooth[0]), float(self.smooth[1]))
        return None


def ui_exclusion_mask(ui_img, tracker, M_ui2img, cam_size, hue_tol=18):
    """Camera-space mask of the places where the PROJECTOR paints the marker colour.
    Without it a red pen stroke or red button would be mistaken for the marker."""
    h, s, v = bgr_to_hsv(ui_img[::2, ::2])
    m = (hue_distance(h, tracker.ref_hue()) <= hue_tol) & (s >= 70) & (v >= 60)
    m8 = cv2.resize(m.astype(np.uint8) * 255, (ui_img.shape[1], ui_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    warped = cv2.warpPerspective(m8, M_ui2img, cam_size)
    return cv2.dilate(warped, np.ones((9, 9), np.uint8))
