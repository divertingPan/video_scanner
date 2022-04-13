[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)

:star: A release on Windows and Ubuntu is updated: https://github.com/divertingPan/video_scanner/releases/tag/v0.3

# video_scanner
Generate the scanning image from a video

![output preview](https://github.com/divertingPan/video_scanner/blob/main/img/1612974206140.png)

![output preview](https://github.com/divertingPan/video_scanner/blob/main/img/1612974211721.png)

![output preview](https://github.com/divertingPan/video_scanner/blob/main/img/1612974217162.png)

How to use the source code: run main.py within a python env

How to use the release software: run the executable 'scan_ver_0.3' in the main folder

## v0.3:

1. The default save directory is the rootpath of the imported video.

2. cv2.imwrite() is replaced by cv2.imencode() to support special characters in the save path. 

3. Add the hint of segment width under 'adaptive' mode to check the moving detection results. 

## v0.2:

Add the adaptive movement detection.

Blog: https://divertingpan.github.io/post/moving_detection/

## v0.1:

Blog: https://divertingpan.github.io/post/train_scanning/


## Hardware Solution
See repo: https://github.com/divertingPan/Line_Scan_Camera
