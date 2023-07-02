# This is a modified algorithm/code based on findings of `Simple video stabilization using OpenCV`
# published on February 20, 2014 by nghiaho12 (http://nghiaho.com/?p=2093)
# modified for video_scanner, on Jun 28, 2023 by divertingPan

"""
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
"""
# import the necessary packages
import cv2
import numpy as np
from collections import deque


class Stabilizer:
    """
    This is an auxiliary class that enables Video Stabilization for vidgear with minimalistic latency, and at the expense
    of little to no additional computational requirements.

    The basic idea behind it is to tracks and save the salient feature array for the given number of frames and then uses
    these anchor point to cancel out all perturbations relative to it for the incoming frames in the queue. This class relies
    heavily on **Threaded Queue mode** for error-free & ultra-fast frame handling.
    """

    def __init__(
        self,
        border_type="black",
        stream=None,
        mask_param=None
    ):

        """
        This constructor method initializes the object state and attributes of the Stabilizer class.

        Parameters:
            border_type (str): changes the extended border type.
            stream (cv2.VideoCapture): for obtaining the first frame and basic information.
            mask_param (List): the position of the mask which disable the disturbing area,
                the item order is [y_top, y_bottom, x_left, x_right].
        """

        # initialize deques for handling input frames and its indexes
        self.__frame_queue = deque(maxlen=5)

        # define and create Adaptive histogram equalization (AHE) object for optimizations
        self.__clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.__previous_gray = None  # it is the first anchor gray frame
        self.__previous_keypoints = None  # it is the first anchor gray frame SIFT keypoints

        self.__frame_height, self.frame_width = 0, 0  # handles width and height of input frames

        # define valid border modes
        border_modes = {
            "black": cv2.BORDER_CONSTANT,
            "reflect": cv2.BORDER_REFLECT,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "wrap": cv2.BORDER_WRAP,
        }
        # choose valid border_mode from border_type
        self.__border_mode = border_modes[border_type]

        # prepare for the first anchor frame
        (grabbed, frame) = stream.read()

        # The mask to detect the keypoint, notice that the mask is meant to disable the moving part, e.g. train
        if mask_param is None:
            mask_param = [0, 0, 0, 0]
        self.y_top = mask_param[0]
        self.y_bottom = mask_param[1]
        self.x_left = mask_param[2]
        self.x_right = mask_param[3]

        self.mask = np.ones(frame.shape[:2], dtype='uint8')
        self.mask[self.y_top:self.y_bottom, self.x_left:self.x_right] = 0

        # get the information of first frame, first frame keypoints, etc.
        self.sift = cv2.SIFT_create()
        self.read_first_frame(frame)

    def read_first_frame(self, frame):
        # Todo: using the first frame of the given video as this previous_gray and keep using it
        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
        previous_gray = self.__clahe.apply(previous_gray)  # optimize gray frame
        self.__previous_keypoints, self.des1 = self.sift.detectAndCompute(previous_gray, mask=self.mask)
        self.__frame_height, self.frame_width = frame.shape[:2]  # save input frame height and width
        self.__previous_gray = previous_gray[:]  # save gray frame clone for further processing

    def stabilize(self, frame):
        """
        This method takes an unstabilized video frame, and returns a stabilized one.

        Parameters:
            frame (numpy.ndarray): inputs unstabilized video frames.
        """
        # check if frame is None
        if frame is None:
            # return if it does
            return

        # start applying transformations
        self.__frame_queue.append(frame)  # save frame to deque
        self.__generate_transformations()  # generate transformations
        return self.__apply_transformations()

    def __generate_transformations(self):
        """
        An internal method that generate previous-to-current transformations [dx,dy,da].
        """
        frame_gray = cv2.cvtColor(
            self.__frame_queue[-1], cv2.COLOR_BGR2GRAY
        )  # retrieve current frame and convert to gray
        frame_gray = self.__clahe.apply(frame_gray)  # optimize it

        try:
            self.curr_kps, des2 = self.sift.detectAndCompute(frame_gray, mask=self.mask)
            # matcher = cv2.BFMatcher()
            matcher = cv2.FlannBasedMatcher()
            raw_matches = matcher.knnMatch(self.des1, des2, k=2)
            self.good_matches = []
            for m1, m2 in raw_matches:
                if m1.distance < 0.85 * m2.distance:
                    self.good_matches.append([m1])

        except cv2.error as e:
            # catch any OpenCV assertion errors and warn user
            print("Video-Frame is too dark to generate any transformations!")
            self.good_matches = None

    def __apply_transformations(self):
        """
        An internal method that applies affine transformation to the given frame
        from previously calculated transformations
        """
        # extract frame and its index from deque
        queue_frame = self.__frame_queue.popleft()

        if len(self.good_matches) > 4:
            ptsA = np.float32([self.__previous_keypoints[m[0].queryIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
            ptsB = np.float32([self.curr_kps[m[0].trainIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
            ransacReprojThreshold = 4
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
            frame_stabilized = cv2.warpPerspective(queue_frame, H,
                                                   (self.__previous_gray.shape[1], self.__previous_gray.shape[0]),
                                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                                   borderMode=self.__border_mode, borderValue=0)
        else:
            frame_stabilized = queue_frame

        # finally return stabilized frame
        return frame_stabilized

    def clean(self):
        """
        Cleans Stabilizer resources
        """
        # check if deque present
        if self.__frame_queue:
            # clear frame deque
            self.__frame_queue.clear()
