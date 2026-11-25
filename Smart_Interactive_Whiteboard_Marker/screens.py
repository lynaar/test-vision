"""All application screens: drawing, lesson, image viewer, calibration helpers, validation, remote learning."""
import numpy as np
import cv2
from ui import (Screen, Button, fill_rect, draw_frame, text, text_fit, TOP_H,
                WHITE, INK, ACCENT, LIGHT, GREY)
from drawing import DrawingLayer, PALETTE, THICKNESSES
import lesson_content


# ------------------------------------------------------------------------- shared pen toolbar
def pen_buttons(layer, x, y, w, h, gap, horizontal):
    """Colours, thicknesses, eraser, undo, clear: used by free drawing and lesson notes."""
    items = []
    for name, col in PALETTE:
        items.append(Button(None, name, lambda c=col: layer.set_color(c),
                            lambda c=col: (not layer.eraser) and layer.pen_color == c, swatch=col))
    for lab, t in zip("SML", THICKNESSES):
        items.append(Button(None, lab, lambda t=t: layer.set_thickness(t), lambda t=t: layer.thickness == t))
    items.append(Button(None, "Erase", layer.toggle_eraser, lambda: layer.eraser))
    items.append(Button(None, "Undo", layer.undo))
    items.append(Button(None, "Clear", layer.clear))
    for i, b in enumerate(items):
        b.rect = (x + i * (w + gap), y, w, h) if horizontal else (x, y + i * (h + gap), w, h)
    return items


def draw_pen_cursor(img, layer, pointer):
    if pointer is not None:
        cv2.circle(img, (int(pointer[0]), int(pointer[1])), layer.radius + 2,
                   (150, 150, 150) if layer.eraser else layer.pen_color, 2, cv2.LINE_AA)


# ------------------------------------------------------------------------- free drawing
class DrawScreen(Screen):
    title = "Free drawing"
    TOOLBAR_W = 170

    def __init__(self, app):
        super().__init__(app)
        self.layer = app.free_layer
        self.buttons = pen_buttons(self.layer, 10, 80, 150, 50, 8, False)

    def in_canvas(self, p):
        return p is not None and p[0] > self.TOOLBAR_W and p[1] > TOP_H

    def on_exit(self):
        self.layer.end_stroke()

    def before_action(self):
        self.layer.end_stroke()

    def update(self, pointer, dt):
        if self.layer.pen_down and self.in_canvas(pointer):
            self.layer.move(pointer)

    def on_left(self, pos):
        if not self.in_canvas(pos):
            return
        if self.layer.pen_down:
            self.layer.end_stroke()
        else:
            self.layer.begin_stroke()
            self.layer.move(pos)

    def draw(self, img, pointer):
        self.layer.compose(img)
        fill_rect(img, (0, 0, self.TOOLBAR_W, self.app.ui_h), LIGHT)
        msg = "PEN DOWN: move to draw - left click to lift" if self.layer.pen_down else "Pen up: left click on the sheet to start drawing"
        text(img, msg, (self.TOOLBAR_W + 30, self.app.ui_h - 25), 0.8, ACCENT if self.layer.pen_down else INK)
        draw_pen_cursor(img, self.layer, pointer if self.in_canvas(pointer) else None)


# ------------------------------------------------------------------------- lesson
class LessonScreen(Screen):
    def __init__(self, app, start=0):
        super().__init__(app)
        self.pages = lesson_content.PAGES
        self.i = start
        self.notes = False
        self._imgs = {}
        self.build()

    def layer(self):
        return self.app.lesson_layers.setdefault(self.i, DrawingLayer(self.app.ui_w, self.app.ui_h))

    def get_title(self):
        return f"Lesson {self.i + 1}/{len(self.pages)}"

    def build(self):
        y = self.app.ui_h - 70
        self.buttons = [Button((15, y, 110, 56), "< Prev", lambda: self.goto(self.i - 1)),
                        Button((135, y, 110, 56), "Next >", lambda: self.goto(self.i + 1)),
                        Button((255, y, 120, 56), "Notes", self.toggle_notes, lambda: self.notes)]
        if self.notes:
            self.buttons += pen_buttons(self.layer(), 390, y, 80, 56, 6, True)

    def toggle_notes(self):
        self.layer().end_stroke()
        self.notes = not self.notes
        self.build()

    def goto(self, i):
        self.layer().end_stroke()
        self.i = max(0, min(len(self.pages) - 1, i))
        self.build()

    def in_area(self, p):
        return p is not None and TOP_H < p[1] < self.app.ui_h - 80

    def on_exit(self):
        self.layer().end_stroke()

    def before_action(self):
        self.layer().end_stroke()

    def update(self, pointer, dt):
        if self.notes and self.layer().pen_down and self.in_area(pointer):
            self.layer().move(pointer)

    def on_left(self, pos):
        if self.notes and self.in_area(pos):
            L = self.layer()
            if L.pen_down:
                L.end_stroke()
            else:
                L.begin_stroke()
                L.move(pos)

    def draw(self, img, pointer):
        page = self.pages[self.i]
        W, H = self.app.ui_w, self.app.ui_h
        text(img, page["title"], (40, 120), 1.25, INK, 3)
        for k, line in enumerate(page["lines"]):
            text(img, line, (50, 180 + k * 44), 0.9, INK, 2)
        if page["image"] is not None:
            if self.i not in self._imgs:
                self._imgs[self.i] = page["image"](W // 2 - 40, H - 230)
            im = self._imgs[self.i]
            x0, y0 = W // 2 + 10, 150
            img[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            draw_frame(img, (x0, y0, im.shape[1], im.shape[0]), GREY, 2)
        if page["annotate"] and not self.notes:
            text(img, "-> press Notes to annotate", (50, H - 100), 0.8, ACCENT)
        self.layer().compose(img)
        if self.notes:
            draw_pen_cursor(img, self.layer(), pointer if self.in_area(pointer) else None)


# ------------------------------------------------------------------------- image viewer + zoom
class ImageScreen(Screen):
    ZOOM_STEP, ZOOM_MAX = 1.4, 12.0

    def __init__(self, app, idx):
        super().__init__(app)
        self.idx = idx
        self.name, self.image = app.images[idx]
        self.area_h = app.ui_h - TOP_H - 80
        self.drag = False
        self._anchor = None
        self.reset_view()
        y = app.ui_h - 70
        labels = [("Zoom +", self.zoom_in), ("Zoom -", self.zoom_out), ("Reset", self.reset_view)]
        x = 15
        self.buttons = []
        for lab, act in labels:
            self.buttons.append(Button((x, y, 120, 56), lab, act)); x += 130
        self.buttons.append(Button((x, y, 120, 56), "Drag", self.toggle_drag, lambda: self.drag)); x += 140
        for lab, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            self.buttons.append(Button((x, y, 90, 56), lab, lambda dx=dx, dy=dy: self.pan(dx, dy))); x += 98

    def get_title(self):
        return f"{self.name}  -  zoom x{self.zoom:.1f}"

    # view = (centre in image pixels, zoom relative to "fit in window")
    def reset_view(self):
        self.zoom = 1.0
        self.cx, self.cy = self.image.shape[1] / 2, self.image.shape[0] / 2
        self.drag = False

    def scale(self):
        ih, iw = self.image.shape[:2]
        return min(self.app.ui_w / iw, self.area_h / ih) * self.zoom

    def clamp(self):
        ih, iw = self.image.shape[:2]
        self.cx = float(np.clip(self.cx, 0, iw)); self.cy = float(np.clip(self.cy, 0, ih))

    def zoom_in(self):
        self.zoom = min(self.ZOOM_MAX, self.zoom * self.ZOOM_STEP)

    def zoom_out(self):
        self.zoom = max(1.0, self.zoom / self.ZOOM_STEP)
        if self.zoom == 1.0:
            self.reset_view()

    def pan(self, dx, dy):
        step = 0.25 * self.app.ui_w / self.scale()
        self.cx += dx * step; self.cy += dy * step
        self.clamp()

    def toggle_drag(self):
        self.drag, self._anchor = not self.drag, None

    def in_area(self, p):
        return p is not None and TOP_H < p[1] < self.app.ui_h - 80

    def update(self, pointer, dt):
        if self.drag and self.in_area(pointer):             # the image follows the pointer
            if self._anchor is None:
                self._anchor = (pointer, self.cx, self.cy)
            else:
                (p0, cx0, cy0) = self._anchor
                s = self.scale()
                self.cx = cx0 - (pointer[0] - p0[0]) / s
                self.cy = cy0 - (pointer[1] - p0[1]) / s
                self.clamp()
        else:
            self._anchor = None

    def draw(self, img, pointer):
        W = self.app.ui_w
        s = self.scale()
        M = np.array([[s, 0, W / 2 - self.cx * s], [0, s, self.area_h / 2 - self.cy * s]], np.float32)
        view = cv2.warpAffine(self.image, M, (W, self.area_h), flags=cv2.INTER_LINEAR, borderValue=(235, 235, 235))
        img[TOP_H:TOP_H + self.area_h] = view
        if self.drag:
            text(img, "DRAG mode: image follows the pointer", (25, TOP_H + 35), 0.8, ACCENT)


# ------------------------------------------------------------------------- instructions / tools
class InstructionScreen(Screen):
    """Shows live instructions; 'mode' tells the app what a click in the Camera window means."""

    def __init__(self, app, title, lines_fn, mode=None, update_fn=None):
        self.title = title
        super().__init__(app)
        self.lines_fn, self.mode, self.update_fn = lines_fn, mode, update_fn

    def on_enter(self):
        self.app.camera_click_mode = self.mode
        if self.mode == "corners":
            self.app.calib_points = []

    def on_exit(self):
        self.app.camera_click_mode = None

    def update(self, pointer, dt):
        if self.update_fn:
            self.update_fn()

    def draw(self, img, pointer):
        H, W = img.shape[:2]
        draw_frame(img, (0, 0, W, H), ACCENT, 6)          # projected border = visible board area
        for k, line in enumerate(self.lines_fn()):
            text(img, line, (80, 200 + k * 55), 0.95, INK, 2)


class ValidationScreen(Screen):
    """Calibration check: put the marker tip on each cross and press NEXT; the error in mm between
    the detected board position and the true cross position is measured."""
    title = "Validate calibration"
    TARGETS = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.5, 0.5)]

    def __init__(self, app):
        super().__init__(app)
        self.k, self.errors = 0, []

    def on_left(self, pos):
        if self.k >= len(self.TARGETS):
            self.app.home()
            return
        if self.app.pointer_board is None:
            self.app.toast("Pointer not detected")
            return
        tx, ty = self.TARGETS[self.k]
        true = np.array([tx * self.app.calib.bw, ty * self.app.calib.bh])
        self.errors.append(float(np.hypot(*(np.array(self.app.pointer_board) - true))))
        self.k += 1
        if self.k == len(self.TARGETS):
            print("Validation errors (mm):", [round(e, 1) for e in self.errors])

    def draw(self, img, pointer):
        W, H = self.app.ui_w, self.app.ui_h
        if self.k < len(self.TARGETS):
            tx, ty = self.TARGETS[self.k]
            c = (int(tx * W), int(ty * H))
            cv2.drawMarker(img, c, ACCENT, cv2.MARKER_CROSS, 60, 4)
            cv2.circle(img, c, 30, ACCENT, 2, cv2.LINE_AA)
            text(img, f"Target {self.k + 1}/{len(self.TARGETS)}: put the marker tip on the cross,", (150, H // 2 - 20), 0.9)
            text(img, "then press NEXT (left click). Right click = skip validation.", (150, H // 2 + 25), 0.9)
        else:
            text(img, f"Mean error: {np.mean(self.errors):.1f} mm   Max: {np.max(self.errors):.1f} mm", (150, H // 2 - 20), 1.1, ACCENT, 3)
            text(img, "Left click to go to the main menu.", (150, H // 2 + 40), 0.9)


class RemoteLearnScreen(Screen):
    title = "Learn remote keys"

    def __init__(self, app):
        super().__init__(app)
        self._saved = False

    def on_enter(self):
        self.app.remote.start_learning()

    def on_exit(self):
        self.app.remote.cancel_learning()

    def update(self, pointer, dt):
        r = self.app.remote
        if r.done and not self._saved:
            self._saved = True
            self.app.cfg["remote"] = r.to_config()
            self.app.save()
            self.app.toast(f"Saved: Previous={sorted(r.right)}  Next={sorted(r.left)}", 6)

    def on_right(self, pos):          # the remote itself cannot be used to leave: Esc / Back button
        pass

    def draw(self, img, pointer):
        r = self.app.remote
        H = self.app.ui_h
        text(img, "(window must have keyboard focus - Esc to cancel)", (80, 160), 0.8, INK)
        if r.done:
            text(img, f"PREVIOUS (right click) codes: {sorted(r.right)}", (80, 260), 0.95, ACCENT)
            text(img, f"NEXT (left click) codes: {sorted(r.left)}", (80, 320), 0.95, ACCENT)
            text(img, "Saved in config.json. Use Back.", (80, 400), 0.9)
        else:
            text(img, r.status(), (80, 280), 1.0, ACCENT, 3)
            text(img, f"last key code received: {r.last_code}", (80, 350), 0.9)
