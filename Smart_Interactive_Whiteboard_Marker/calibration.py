"""Calibration and coordinate systems.

Three coordinate systems are used:
  * IMAGE  (pixels)  : camera image, origin top-left.
  * BOARD  (mm)      : WORLD frame attached to the upper-left corner of the board,
                       x to the right, y downwards, units = millimetres.
  * UI     (pixels)  : projected interface; the projector fills the board, so
                       UI = BOARD * (ui_size / board_size).

The mapping IMAGE -> BOARD is a homography (plane to plane projective transform) estimated with the
Direct Linear Transform (DLT) implemented below with numpy only (SVD for the linear system).
"""
import numpy as np
import cv2


def apply_h(H, pts):
    """Apply a 3x3 homography to an (N,2) array of points."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def _normalisation(pts):
    """Hartley normalisation: centroid at 0, mean distance sqrt(2) (better conditioned SVD)."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def homography_dlt(src, dst):
    """Homography H such that dst ~ H * src, from N >= 4 correspondences."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    Ts, Td = _normalisation(src), _normalisation(dst)
    s, d = apply_h(Ts, src), apply_h(Td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows))
    H = np.linalg.inv(Td) @ vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex(p):
    signs = []
    for i in range(4):
        a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        signs.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


class BoardCalibration:
    def __init__(self, board_size_mm, ui_size):
        self.bw, self.bh = board_size_mm
        self.uw, self.uh = ui_size
        self.corners = None
        self.H = self.Hinv = None
        self._board_pts = np.array([[0, 0], [self.bw, 0], [self.bw, self.bh], [0, self.bh]], float)

    @property
    def ready(self):
        return self.H is not None

    def set_corners(self, corners_img, cam_size):
        """corners_img: 4 clicked points in the order TL, TR, BR, BL. Raises ValueError if invalid."""
        c = np.array(corners_img, float).reshape(4, 2)
        if not is_convex(c):
            raise ValueError("Corners not convex / wrong order (TL, TR, BR, BL)")
        if polygon_area(c) < 0.05 * cam_size[0] * cam_size[1]:
            raise ValueError("Board too small in the camera image")
        self.corners = c
        self.H = homography_dlt(c, self._board_pts)
        self.Hinv = np.linalg.inv(self.H)

    # ---- conversions -------------------------------------------------------------------
    def img_to_board(self, p):
        return apply_h(self.H, [p])[0]

    def board_to_img(self, b):
        return apply_h(self.Hinv, [b])[0]

    def board_to_ui(self, b):
        return np.array([b[0] * self.uw / self.bw, b[1] * self.uh / self.bh])

    def ui_to_board(self, u):
        return np.array([u[0] * self.bw / self.uw, u[1] * self.bh / self.uh])

    def ui_to_img_matrix(self):
        """3x3 matrix sending projected-UI pixels to camera pixels (used to warp UI masks)."""
        S = np.diag([self.bw / self.uw, self.bh / self.uh, 1.0])
        return self.Hinv @ S

    # ---- validation --------------------------------------------------------------------
    def round_trip_error_mm(self):
        """Corners -> board -> corners; should be ~0. Detects numerical problems."""
        mapped = apply_h(self.H, self.corners)
        return float(np.abs(mapped - self._board_pts).max())

    def roi_mask(self, shape):
        """Camera-image mask of the board area (marker is searched only there)."""
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [self.corners.astype(np.int32)], 255)   # polygon rasterisation
        return m
