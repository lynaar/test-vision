"""Headless tests (no camera, no window):  python selftest.py"""
import numpy as np
import cv2
from calibration import homography_dlt, apply_h, BoardCalibration
from pointer_detection import PointerTracker, bgr_to_hsv
from remote_input import RemoteInput
from config import load_config
from app import App
import screens

# 1. homography recovered from 4 correspondences
H = np.array([[1.1, 0.2, 30], [0.05, 0.9, 12], [1e-4, 2e-4, 1]])
src = np.array([[100, 80], [1100, 120], [1180, 650], [60, 600]], float)
Hest = homography_dlt(src, apply_h(H, src))
assert np.abs(apply_h(Hest, src) - apply_h(H, src)).max() < 1e-6
print("homography OK")

# 2. HSV conversion matches OpenCV
rng = np.random.default_rng(0)
im = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
h, s, v = bgr_to_hsv(im)
ref = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
dh = np.abs(h - ref[..., 0].astype(int)); dh = np.minimum(dh, 180 - dh)      # hue is circular
assert dh.max() <= 2 and np.abs(s - ref[..., 1]).max() <= 2 and (v == ref[..., 2]).all()
print("HSV OK")

# 3. synthetic scene: board = skewed quad, red marker at a known board position
cam_w, cam_h = 1280, 720
corners = np.array([[200, 100], [1050, 140], [1000, 620], [150, 560]], float)
cal = BoardCalibration([1200.0, 675.0], [1280, 720])
cal.set_corners(corners, (cam_w, cam_h))
target_board = np.array([300.0, 200.0])
tx, ty = cal.board_to_img(target_board)
tracker = PointerTracker()
roi = cal.roi_mask((cam_h, cam_w))
res = None
for _ in range(6):
    f = np.full((cam_h, cam_w, 3), 200, np.uint8)
    cv2.circle(f, (int(tx), int(ty)), 14, (30, 30, 230), -1)             # marker
    cv2.circle(f, (60, 60), 20, (30, 30, 230), -1)                       # red object OUTSIDE the board
    cv2.ellipse(f, (600, 300), (60, 25), 0, 0, 360, (120, 130, 220), -1)  # skin-like blob
    _, res = tracker.detect(f, roi)
assert res is not None, "marker not detected"
err = np.hypot(*(cal.img_to_board(res) - target_board))
print(f"marker mapped to board with error {err:.2f} mm")
assert err < 5

# 4. remote learning
cfg = load_config()
r = RemoteInput(cfg)
r.start_learning()
r.poll(2162688, 0.0); r.poll(2162689, 0.1); r.poll(-1, 1.0)
r.poll(2228224, 1.5); r.poll(-1, 2.5)
assert r.done and 2162688 in r.right and 2228224 in r.left
print("remote learning OK:", r.to_config())
r2 = RemoteInput(cfg)
assert r2.poll(2228224, 10.0) == "left" and r2.poll(2162688, 20.0) == "right"

# 5. UI walk-through in simulation
app = App(cfg, sim=True)
app.set_sim_pointer(640, 300)
for _ in range(3):
    app.render()
app.push(screens.DrawScreen(app))
app.click("left")
for x in range(400, 800, 20):
    app.set_sim_pointer(x, 300 + (x % 60)); app.render()
app.click("left")
assert app.free_layer.mask.sum() > 500
app.home()
app.push(screens.LessonScreen(app))
for _ in range(7):
    app.stack[-1].goto(app.stack[-1].i + 1); app.render()
app.home(); app.push(app.image_menu()); app.render()
app.push(screens.ImageScreen(app, 0))
sc = app.stack[-1]
for _ in range(5): sc.zoom_in()
sc.pan(1, 0); app.render(); sc.reset_view(); app.render()
app.home(); app.push(app.tools_menu()); app.render()
app.push(screens.ValidationScreen(app)); app.render()
app.push(screens.RemoteLearnScreen(app)); app.render()
app.home()
cv2.imwrite("/tmp/menu.png", app.render())
print("UI OK")
