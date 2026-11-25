"""Configuration file handling (JSON). Everything that must survive a restart is stored here:
board size, camera corners (calibration), marker colour and the remote key codes."""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")

# Key codes as returned by cv2.waitKeyEx(). Most presentation remotes send Page Down / Page Up
# (or Right / Left arrows). The values differ between Windows and Linux, so both are listed.
# The real codes of YOUR remote are learned in Tools > Learn remote keys and stored in config.json.
DEFAULTS = {
    "camera_index": 0,
    "camera_size": [1280, 720],
    "ui_size": [1280, 720],                  # size of the projected interface (pixels)
    "board_size_mm": [1200.0, 675.0],        # physical size of the projected board area
    "corners_img": None,                     # 4 board corners in the camera image (TL, TR, BR, BL)
    "marker_hsv": None,                      # calibrated marker colour, None = bright red
    "remote": {
        "left": [13, 2228224, 65366, 2555904, 65363],    # NEXT     -> left click  (Enter / PgDn / Right)
        "right": [2162688, 65365, 2424832, 65361],   # PREVIOUS -> right click (PgUp / Left)
    },
}


def load_config():
    cfg = copy.deepcopy(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg.update(json.load(f))
        except (OSError, ValueError):
            print("config.json unreadable, using defaults")
    # always keep the default remote codes (e.g. Enter = 13) even if an old config.json lacks them
    for side in ("left", "right"):
        cfg["remote"][side] = sorted(set(cfg["remote"][side]) | set(DEFAULTS["remote"][side]))
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)