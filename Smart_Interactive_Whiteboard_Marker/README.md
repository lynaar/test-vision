# Interactive whiteboard (camera + projector + presentation remote)

## Install / run
    pip install opencv-python numpy
    python app.py                # real setup (camera + projector)
    python app.py --sim          # no hardware: mouse = pointer, left/right mouse button = remote
    python selftest.py           # headless tests (homography, HSV, marker->board, remote, UI)

## Set-up procedure (also the demo order)
1. Drag the **Board** window onto the projector, press **F** (fullscreen).
2. **Calibration** (starts automatically the first time, or Tools > Calibrate / key C):
   click the 4 board corners in the **Camera** window: TL, TR, BR, BL. The projector must fill the board.
   Then the 5-cross **validation** measures the error in mm (Tools > Validate calibration).
3. **Marker colour** (Tools > Marker colour / K): click on the marker in the Camera window (default = vivid red).
4. **Remote** (Tools > Learn remote keys / R): press PREVIOUS then NEXT. Codes saved in config.json
   and logged in remote_keys.log. Defaults: NEXT = Page Down / Right (left click), PREVIOUS = Page Up / Left (right click).
5. Use: point with the marker, NEXT = left click (press / pen down-up), PREVIOUS = right click (back one level).

## Files
| file | role |
|---|---|
| camera.py | acquisition thread (latest frame only) |
| calibration.py | DLT homography (own SVD-based code), coordinate systems, validation, ROI |
| pointer_detection.py | own BGR->HSV, thresholds, erosion/dilation, centroid, tracking, smoothing, projector-colour exclusion |
| remote_input.py | key codes (cv2.waitKeyEx), learning mode, debounce, click mapping |
| drawing.py | pen/eraser primitives (disc stamping), colours, thickness, undo, clear |
| ui.py / screens.py | widgets, menus (main + submenus), drawing, lesson, image viewer + zoom, validation |
| lesson_content.py | lesson pages and generated images |
| app.py | main loop, navigation, glue |

## Predefined functions used (and why)
cv2.VideoCapture/imshow/waitKeyEx (I/O) - cv2.connectedComponentsWithStats (labelling; centroid is our own) -
cv2.fillPoly (ROI rasterisation) - cv2.warpPerspective/warpAffine/resize (image resampling: UI mask warp, zoom) -
cv2.putText/circle/polylines (text and overlay drawing) - np.linalg.svd (least-squares solution inside our DLT).

## Known limits
Marker hidden by hand/body; projected content of the marker colour is masked (marker invisible over it);
strong ambient light change -> recalibrate colour; remote needs keyboard focus on a window; projector must fill the board.
