# Smart Interactive Whiteboard - Marker Version

This version controls the board with a coloured marker, not hand tracking.

## Recommended marker

Use a bright red marker tip or a small vivid red object. Avoid a pale, dark, reflective, or transparent marker.

## Start

1. Install the dependencies: `pip install -r requirements.txt`
2. Start the real camera version: `python app.py`
3. At first launch, click the four board corners in the Camera window.
4. Press `K`, then click the marker tip in the Camera window to learn its colour.
5. Point the marker at the board. The Board window shows the detected position as a circle.

## Demonstration without hardware

Run `python app.py --sim`. The mouse replaces the marker; left and right mouse buttons replace the remote.

## Useful keys

- `C`: calibrate board corners
- `K`: learn marker colour
- `X`: restore the default vivid red marker colour
- `V`: validate calibration
- `R`: learn remote buttons
- `F`: fullscreen Board window
- `Q`: quit

The marker must stay visible to the camera. Recalibrate its colour if lighting changes significantly.
