# video_scanner
Generate the scanning image from a video

![output preview](https://github.com/divertingPan/video_scanner/blob/main/img/1612974206140.png)

![output preview](https://github.com/divertingPan/video_scanner/blob/main/img/1612974211721.png)

![output preview](https://github.com/divertingPan/video_scanner/blob/main/img/1612974217162.png)

Just run main.py within python envs

## v0.3:

The default save directory is the rootpath of the imported video.

cv2.imwrite() is replaced by cv2.imencode() to support special characters in the save path. 

## v0.2:

Add the adaptive movement detection.

Blog: https://divertingpan.github.io/post/moving_detection/

## v0.1:

Blog: https://divertingpan.github.io/post/train_scanning/


## Hardware Solution
See repo: https://github.com/divertingPan/Line_Scan_Camera
