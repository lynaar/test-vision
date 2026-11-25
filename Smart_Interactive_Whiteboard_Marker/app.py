"""Interactive whiteboard: camera + projector + presentation remote.

Run:  python app.py [--camera N] [--sim] [--fullscreen]
  --sim : no camera, the mouse plays the pointer and its buttons play the remote (for development).

Windows:  "Board"  = projected interface (drag it on the projector, press F for fullscreen)
          "Camera" = what the camera sees + detection overlay (calibration clicks happen here)
Keyboard shortcuts (focus on a window): F fullscreen, D marker debug view, C calibrate,
V validate, K marker colour, X forget colour, R learn remote, Q quit.

Data flow per frame:
  Camera -> PointerTracker (image px) -> BoardCalibration (board mm, UI px) -> pointer
  Remote key -> RemoteInput -> 'left'/'right' -> click at the pointer -> current Screen
"""
import argparse
import os
import time
import glob
import cv2
import numpy as np

from config import load_config, save_config, HERE
from camera import Camera
from calibration import BoardCalibration
from pointer_detection import PointerTracker, ui_exclusion_mask
from remote_input import RemoteInput
from drawing import DrawingLayer
from ui import MenuScreen, text, ACCENT, WHITE
from screens import (DrawScreen, LessonScreen, ImageScreen, InstructionScreen,
                     ValidationScreen, RemoteLearnScreen)
import lesson_content


class App:
    POINTER_GRACE = 0.6      # s: a click still uses the last pointer position for this long

    def __init__(self, cfg, sim=False):
        self.cfg, self.sim = cfg, sim
        self.ui_w, self.ui_h = cfg["ui_size"]
        self.cam_size = tuple(cfg["camera_size"])
        self.calib = BoardCalibration(cfg["board_size_mm"], cfg["ui_size"])
        self.tracker = PointerTracker(cfg["marker_hsv"])
        self.remote = RemoteInput(cfg)
        self.roi = None
        self.exclusion = None
        self.frame_no = 0
        self.last_frame = None
        self.last_ui = None
        self.marker_mask = None
        self.cam_center = None
        # pointer state
        self.pointer_ui = None
        self.pointer_board = None
        self.sim_pointer = None
        self._pointer_time = 0.0
        # camera-window click modes
        self.camera_click_mode = None
        self.calib_points = []
        self.color_click = None
        # content
        self.free_layer = DrawingLayer(self.ui_w, self.ui_h)
        self.lesson_layers = {}
        self.images = self._load_images()
        self.toasts = []
        self.stack = []
        self.quit = False
        if cfg.get("corners_img"):
            try:
                self.calib.set_corners(cfg["corners_img"], self.cam_size)
                self.roi = self.calib.roi_mask((self.cam_size[1], self.cam_size[0]))
            except ValueError:
                pass
        self.push(self.main_menu())

    # ---------------------------------------------------------------- content / menus
    def _load_images(self):
        imgs = lesson_content.builtin_images()
        for p in sorted(glob.glob(os.path.join(HERE, "images", "*")))[:6]:
            im = cv2.imread(p)
            if im is not None:
                s = 2000 / max(im.shape[:2])
                if s < 1:
                    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                imgs.append((os.path.basename(p), im))
        return imgs

    def main_menu(self):
        return MenuScreen(self, "Interactive board - main menu", [
            ("Lesson", lambda: self.push(LessonScreen(self))),
            ("Free drawing", lambda: self.push(DrawScreen(self))),
            ("Images and zoom", lambda: self.push(self.image_menu())),
            ("Tools", lambda: self.push(self.tools_menu())),
            ("Quit", self.request_quit)], nav=False)

    def image_menu(self):
        items = [(n, lambda i=i: self.push(ImageScreen(self, i))) for i, (n, _) in enumerate(self.images)]
        return MenuScreen(self, "Images (submenu)", items)

    def tools_menu(self):
        return MenuScreen(self, "Tools (submenu)", [
            ("Calibrate board corners", self.start_calibration),
            ("Validate calibration", lambda: self.push(ValidationScreen(self))),
            ("Marker colour", self.start_color_calibration),
            ("Forget marker colour", self.forget_color),
            ("Learn remote keys", lambda: self.push(RemoteLearnScreen(self))),
            ("Reset application", self.reset)])

    # ---------------------------------------------------------------- navigation
    def push(self, screen):
        if self.stack:
            self.stack[-1].on_exit()
        self.stack.append(screen)
        screen.on_enter()

    def back(self):
        if len(self.stack) > 1:
            self.stack.pop().on_exit()
            self.stack[-1].on_enter()

    def home(self):
        while len(self.stack) > 1:
            self.stack.pop().on_exit()
        self.stack[-1].on_enter()

    def request_quit(self):
        self.quit = True

    def reset(self):
        self.free_layer = DrawingLayer(self.ui_w, self.ui_h)
        self.lesson_layers = {}
        self.home()
        self.toast("Application reset")

    def toast(self, msg, secs=3.0):
        self.toasts.append((msg, time.time() + secs))

    def save(self):
        save_config(self.cfg)

    # ---------------------------------------------------------------- calibration / colour
    def start_calibration(self):
        names = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]

        def lines():
            n = len(self.calib_points)
            return ["CALIBRATION", "In the CAMERA window, click the 4 corners of the board:",
                    "TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT",
                    f"Next corner: {names[n] if n < 4 else '-'}   ({n}/4)"]
        self.push(InstructionScreen(self, "Calibration", lines, "corners", self._check_calibration))

    def _check_calibration(self):
        if len(self.calib_points) == 4:
            self.finish_calibration()

    def finish_calibration(self):
        pts, self.calib_points = list(self.calib_points), []
        shape = self.last_frame.shape if self.last_frame is not None else (self.cam_size[1], self.cam_size[0], 3)
        try:
            self.calib.set_corners(pts, (shape[1], shape[0]))
        except ValueError as e:
            self.toast(str(e), 5)
            return
        self.roi = self.calib.roi_mask(shape)
        self.cfg["corners_img"] = [[float(x), float(y)] for x, y in self.calib.corners]
        self.save()
        self.tracker.reset()
        self.toast(f"Calibration OK (round-trip error {self.calib.round_trip_error_mm():.3f} mm)")
        self.back()
        self.push(ValidationScreen(self))

    def start_color_calibration(self):
        self.color_click = None
        self.push(InstructionScreen(
            self, "Marker colour",
            lambda: ["MARKER COLOUR", "Hold the marker in front of the board,",
                     "then click on it in the CAMERA window.", f"Current: {self.tracker.hsv_ref or 'vivid red (default)'}"],
            "color", self._check_color))

    def _check_color(self):
        if self.color_click is not None and self.last_frame is not None:
            if self.tracker.calibrate(self.last_frame, self.color_click):
                self.cfg["marker_hsv"] = list(self.tracker.hsv_ref)
                self.save()
                self.toast(f"Marker colour learned (HSV {self.tracker.hsv_ref})")
            self.color_click = None
            self.back()

    def forget_color(self):
        self.tracker.forget_color()
        self.cfg["marker_hsv"] = None
        self.save()
        self.toast("Marker colour: default (vivid red)")

    # ---------------------------------------------------------------- per-frame processing
    def process_frame(self, frame, now=None):
        """Detect the marker and update the pointer (UI px and board mm)."""
        now = time.time() if now is None else now
        self.last_frame = frame
        self.frame_no += 1
        if self.sim or not self.calib.ready:
            self.cam_center = None
            return
        if self.roi is None or self.roi.shape != frame.shape[:2]:
            self.roi = self.calib.roi_mask(frame.shape)
        if self.last_ui is not None and self.frame_no % 6 == 1:      # refresh projector-colour mask
            self.exclusion = ui_exclusion_mask(self.last_ui, self.tracker, self.calib.ui_to_img_matrix(),
                                               (frame.shape[1], frame.shape[0]))
        self.marker_mask, self.cam_center = self.tracker.detect(frame, self.roi, self.exclusion)
        if self.cam_center is not None:
            b = self.calib.img_to_board(self.cam_center)
            u = self.calib.board_to_ui(b)
            self.pointer_board = (float(b[0]), float(b[1]))
            self.pointer_ui = (float(np.clip(u[0], 0, self.ui_w - 1)), float(np.clip(u[1], 0, self.ui_h - 1)))
            self._pointer_time = now
        elif now - self._pointer_time > self.POINTER_GRACE:
            self.pointer_ui = self.pointer_board = None

    def set_sim_pointer(self, x, y):
        self.sim_pointer = (x, y)
        self.pointer_ui = (float(x), float(y))
        b = self.calib.ui_to_board((x, y))
        self.pointer_board = (float(b[0]), float(b[1]))
        self._pointer_time = time.time()

    def click(self, button):
        """Remote button -> click at the current pointer position."""
        if self.pointer_ui is None:
            self.toast("Pointer not detected - click ignored", 1.5)
            return
        self.stack[-1].click(self.pointer_ui, button)

    def handle_key(self, key, now=None):
        now = time.time() if now is None else now
        if key == 27 and self.remote.learning:          # Esc cancels remote learning
            self.back()
            return
        action = self.remote.poll(key, now)
        if action:
            self.click(action)
            return
        ch = chr(key) if 0 <= key < 128 else ""
        if self.remote.learning or not ch:
            return
        if ch == "q": self.quit = True
        elif ch == "c": self.start_calibration()
        elif ch == "v": self.push(ValidationScreen(self))
        elif ch == "k": self.start_color_calibration()
        elif ch == "x": self.forget_color()
        elif ch == "r": self.push(RemoteLearnScreen(self))

    # ---------------------------------------------------------------- rendering
    def render(self):
        self.stack[-1].update(self.pointer_ui, 0.03)
        img = self.stack[-1].render(self.pointer_ui)
        p = self.pointer_ui
        if p is not None:                               # visible pointer
            cv2.circle(img, (int(p[0]), int(p[1])), 22, ACCENT, 3, cv2.LINE_AA)
            cv2.circle(img, (int(p[0]), int(p[1])), 4, ACCENT, -1, cv2.LINE_AA)
        info = "pointer: x=%.0f y=%.0f mm" % self.pointer_board if self.pointer_board else "pointer: none"
        text(img, info, (self.ui_w - 330, self.ui_h - 12), 0.55, (130, 130, 130), 1)
        now = time.time()
        self.toasts = [t for t in self.toasts if t[1] > now]
        for k, (msg, _) in enumerate(self.toasts[-3:]):
            text(img, msg, (25, self.ui_h - 95 - 32 * k), 0.75, (0, 0, 180), 2)
        self.last_ui = img
        return img

    def camera_view(self, frame):
        view = frame.copy() if frame is not None else np.zeros((self.cam_size[1], self.cam_size[0], 3), np.uint8)
        if self.calib.corners is not None:
            cv2.polylines(view, [self.calib.corners.astype(np.int32)], True, (0, 255, 0), 3)
        for i, (x, y) in enumerate(self.calib_points):
            cv2.circle(view, (int(x), int(y)), 8, (0, 255, 255), -1)
            cv2.putText(view, str(i + 1), (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if self.cam_center is not None:
            c = (int(self.cam_center[0]), int(self.cam_center[1]))
            cv2.circle(view, c, 18, (0, 255, 0), 3)
            cv2.putText(view, "MARKER", (c[0] + 22, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        status = "SIMULATION (mouse)" if self.sim else ("calibrated" if self.calib.ready else "NOT calibrated: press C")
        cv2.putText(view, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return view

    # ---------------------------------------------------------------- main loop
    def run(self, cam_index=None, fullscreen=False):
        cam = None
        if not self.sim:
            cam = Camera(cam_index if cam_index is not None else self.cfg["camera_index"], self.cam_size)
            if not cam.ok:
                print("Camera not available -> simulation mode (mouse).")
                self.sim, cam = True, None
        cv2.namedWindow("Board", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Board", self.ui_w, self.ui_h)
        cv2.namedWindow("Camera")
        if fullscreen:
            cv2.setWindowProperty("Board", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        def on_board_mouse(event, x, y, flags, _):
            if not self.sim:
                return
            self.set_sim_pointer(x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.click("left")
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.click("right")

        def on_cam_mouse(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.camera_click_mode == "corners" and len(self.calib_points) < 4:
                    self.calib_points.append((x, y))
                elif self.camera_click_mode == "color":
                    self.color_click = (x, y)

        cv2.setMouseCallback("Board", on_board_mouse)
        cv2.setMouseCallback("Camera", on_cam_mouse)
        if not self.sim and not self.calib.ready:
            self.start_calibration()
        show_mask, board_full = False, fullscreen

        while not self.quit:
            frame = cam.read() if cam else None
            if frame is not None:
                self.process_frame(frame)
            cv2.imshow("Board", self.render())
            cv2.imshow("Camera", self.camera_view(frame))
            if show_mask and self.marker_mask is not None:
                cv2.imshow("Marker mask", self.marker_mask)
            key = cv2.waitKeyEx(1)
            if key in (ord("f"), ord("d")) and not self.remote.learning:
                if key == ord("f"):
                    board_full = not board_full
                    cv2.setWindowProperty("Board", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if board_full else cv2.WINDOW_NORMAL)
                else:
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow("Marker mask")
                continue
            if key != -1:
                self.handle_key(key)
        if cam:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--sim", action="store_true", help="mouse simulates pointer + remote")
    ap.add_argument("--fullscreen", action="store_true")
    a = ap.parse_args()
    App(load_config(), sim=a.sim).run(a.camera, a.fullscreen)
