"""Interface toolkit: drawing helpers, Button, Screen base class, MenuScreen.
The interface is white with dark elements (a projector shows black as 'no light', so a white
UI looks like a normal board). Accent colour is blue: far from the default red marker."""
import math
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, INK, ACCENT = (255, 255, 255), (50, 50, 50), (170, 90, 30)
LIGHT, HOVER, GREY = (240, 232, 225), (250, 215, 170), (190, 190, 190)
TOP_H = 70


def fill_rect(img, rect, color):
    x, y, w, h = rect
    img[max(0, y):y + h, max(0, x):x + w] = color


def draw_frame(img, rect, color, t=2):
    x, y, w, h = rect
    for r in ((x, y, w, t), (x, y + h - t, w, t), (x, y, t, h), (x + w - t, y, t, h)):
        fill_rect(img, r, color)


def text(img, s, org, scale=0.9, color=INK, thick=2):
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, color, thick, cv2.LINE_AA)


def text_fit(img, s, rect, color, base=0.9, thick=2):
    """Centered text, scaled down if it does not fit in rect."""
    x, y, w, h = rect
    (tw, th), _ = cv2.getTextSize(s, FONT, 1.0, thick)
    sc = min(base, (w - 14) / max(tw, 1), (h - 10) / max(th, 1))
    text(img, s, (x + (w - tw * sc) / 2, y + (h + th * sc) / 2), sc, color, thick)


class Button:
    def __init__(self, rect, label, action, toggled=None, swatch=None):
        self.rect, self.label, self.action = rect, label, action
        self.toggled, self.swatch = toggled, swatch

    def contains(self, p):
        x, y, w, h = self.rect
        return p is not None and x <= p[0] < x + w and y <= p[1] < y + h

    def draw(self, img, hover):
        on = bool(self.toggled and self.toggled())
        if self.swatch is not None:
            fill_rect(img, self.rect, self.swatch)
            draw_frame(img, self.rect, ACCENT if (on or hover) else GREY, 6 if on else 3)
            return
        fill_rect(img, self.rect, ACCENT if on else (HOVER if hover else LIGHT))
        draw_frame(img, self.rect, ACCENT, 2)
        text_fit(img, self.label, self.rect, WHITE if on else INK)


class Screen:
    """A screen = buttons + content. Left click presses the button under the pointer or calls
    on_left(); right click goes back one level (override on_right to change)."""
    show_nav = True
    title = ""

    def __init__(self, app):
        self.app = app
        self.buttons = []
        self.nav = []
        if self.show_nav:
            W = app.ui_w
            self.nav = [Button((W - 130, 10, 120, 50), "Home", app.home),
                        Button((W - 260, 10, 120, 50), "Back", app.back)]

    def get_title(self):
        return self.title

    def on_enter(self): pass
    def on_exit(self): pass
    def update(self, pointer, dt): pass
    def before_action(self): pass
    def on_left(self, pos): pass
    def on_right(self, pos): self.app.back()
    def draw(self, img, pointer): pass

    def render(self, pointer):
        img = np.full((self.app.ui_h, self.app.ui_w, 3), 255, np.uint8)
        self.draw(img, pointer)
        text(img, self.get_title(), (25, 48), 1.1, ACCENT, 3)
        for b in self.nav + self.buttons:
            b.draw(img, b.contains(pointer))
        return img

    def click(self, pos, button):
        if button == "left":
            for b in self.nav + self.buttons:
                if b.contains(pos):
                    self.before_action()
                    b.action()
                    return
            self.on_left(pos)
        else:
            self.on_right(pos)


class MenuScreen(Screen):
    """Generic menu: list of (label, action). Actions may push another MenuScreen (submenus)."""

    def __init__(self, app, title, items, nav=True):
        self.title, self.show_nav = title, nav
        super().__init__(app)
        n = len(items)
        cols = 1 if n <= 5 else 2
        rows = math.ceil(n / cols)
        bw, bh, gap = (520 if cols == 1 else 470), 80, 22
        total_h = rows * bh + (rows - 1) * gap
        y0 = max(110, (app.ui_h - total_h) // 2 + 30)
        x0 = (app.ui_w - (cols * bw + (cols - 1) * 40)) // 2
        for i, (label, action) in enumerate(items):
            c, r = divmod(i, rows)
            self.buttons.append(Button((x0 + c * (bw + 40), y0 + r * (bh + gap), bw, bh), label, action))
