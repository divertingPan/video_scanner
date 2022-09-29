import os
import cv2
import math
import time
from threading import Thread
from queue import Queue
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
import tkinter.font as tf
from tkinter import ttk
from tkinter import filedialog


class FileVideoStream:
    """
    https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
    """
    def __init__(self, path, queue_size=16):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)
            else:
                # avoiding GIL problem
                time.sleep(0.001)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # def more(self):
    #     # return True if there are still frames in the queue
    #     return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def running(self):
        # indicate that the thread is still running
        return not self.stopped


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.total_frames = None
        self.label_status = None
        self.progressbar = None
        self.label_segment = None
        self.text_x_right = None
        self.text_x_left = None
        self.text_y_bottom = None
        self.text_y_top = None
        self.text_adaptive_sensitivity = None
        self.text_adaptive_start = None
        self.v_left_right = None
        self.text_width = None
        self.v_adaptive_control = None
        self.text_split_width = None
        self.text_position = None
        self.label_video_attr = None
        self.save_path = None
        self.video_path = None
        self.frame_width = None
        self.frame_height = None
        self.frame_right_1 = None
        self.frame_right_2 = None
        self.frame_right_3 = None
        self.vc = None
        self.canvas = None
        self.fig = None

        self.master = master
        self.font = tf.Font(size=12)
        self.left_canvas()
        self.right_panel()
        self.video_args_text()
        self.pos_split_args_ctrl()
        self.video_clip_interval_ctrl()
        self.start_and_status()

    """
        set the panels and widgets on GUI
    """

    def left_canvas(self):
        frame_left = tk.Frame(self.master)
        frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        self.fig = Figure(figsize=(8, 4), dpi=72)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_left)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        toolbar = NavigationToolbar2Tk(self.canvas, frame_left)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        scrollbar_display = tk.Scale(frame_left, orient=tk.HORIZONTAL, from_=0, to=1000,
                                     resolution=1, command=self.display_frames)
        scrollbar_display.pack(fill=tk.X)

    def right_panel(self):
        frame_right = tk.Frame(self.master)
        frame_right.pack(side=tk.LEFT, padx=10, expand=tk.YES)
        self.frame_right_1 = tk.Frame(frame_right)
        self.frame_right_1.pack()
        self.frame_right_2 = tk.Frame(frame_right)
        self.frame_right_2.pack()
        self.frame_right_3 = tk.Frame(frame_right)
        self.frame_right_3.pack()

    def video_args_text(self):
        self.video_path = tk.StringVar()
        self.video_path.set('... select a video ...')
        label_video_path = tk.Label(self.frame_right_1, textvariable=self.video_path, font=self.font)
        label_video_path.pack()
        bt_open_video = tk.Button(self.frame_right_1, text='open a video',
                                  command=self.open_video, font=self.font)
        bt_open_video.pack()

        self.save_path = tk.StringVar()
        self.save_path.set('... select save dir ...')
        label_save_path = tk.Label(self.frame_right_1, textvariable=self.save_path, font=self.font)
        label_save_path.pack()
        bt_select_save_dir = tk.Button(self.frame_right_1, text='save dir',
                                       command=self.select_save_dir, font=self.font)
        bt_select_save_dir.pack()

        self.label_video_attr = tk.Label(self.frame_right_1, font=self.font, text='... ......... ...\n'
                                                                                  '... info area ...\n'
                                                                                  '... ......... ...')
        self.label_video_attr.pack()

    def pos_split_args_ctrl(self):
        label_position = tk.Label(self.frame_right_1, text='position:', font=self.font)
        label_position.pack()
        self.text_position = tk.Entry(self.frame_right_1, font=self.font)
        self.text_position.pack()

        label_split_width = tk.Label(self.frame_right_1, text='split_width:', font=self.font)
        label_split_width.pack()
        self.text_split_width = tk.Entry(self.frame_right_1, font=self.font)
        self.text_split_width.pack()

    def video_clip_interval_ctrl(self):
        self.v_adaptive_control = tk.IntVar()

        # manual panel
        frame_manual = tk.Frame(self.frame_right_2)
        frame_manual.pack(side=tk.LEFT, expand=tk.YES)

        radio_manual_control = tk.Radiobutton(frame_manual, text='manual',
                                              variable=self.v_adaptive_control, value=0, font=self.font)
        radio_manual_control.pack()

        label_width = tk.Label(frame_manual, text='width:', font=self.font)
        label_width.pack()
        self.text_width = tk.Entry(frame_manual, font=self.font)
        self.text_width.pack()

        self.v_left_right = tk.IntVar()
        radio_left_to_right = tk.Radiobutton(frame_manual, text='left to right',
                                             variable=self.v_left_right, value=1, font=self.font)
        radio_left_to_right.pack()
        radio_right_to_left = tk.Radiobutton(frame_manual, text='right to left',
                                             variable=self.v_left_right, value=0, font=self.font)
        radio_right_to_left.pack()

        # adaptive panel
        frame_adaptive = tk.Frame(self.frame_right_2)
        frame_adaptive.pack(side=tk.RIGHT, padx=10, expand=tk.YES)

        radio_adaptive_control = tk.Radiobutton(frame_adaptive, text='adaptive',
                                                variable=self.v_adaptive_control, value=1, font=self.font)
        radio_adaptive_control.pack()

        label_adaptive_start = tk.Label(frame_adaptive, text='start frame:', font=self.font)
        label_adaptive_start.pack()
        self.text_adaptive_start = tk.Entry(frame_adaptive, font=self.font)
        self.text_adaptive_start.pack()

        label_adaptive_sensitivity = tk.Label(frame_adaptive, text='sensitivity:', font=self.font)
        label_adaptive_sensitivity.pack()
        self.text_adaptive_sensitivity = tk.Entry(frame_adaptive, font=self.font)
        self.text_adaptive_sensitivity.pack()

        label_y_top = tk.Label(frame_adaptive, text='mask y_top:', font=self.font)
        label_y_top.pack()
        self.text_y_top = tk.Entry(frame_adaptive, font=self.font)
        self.text_y_top.pack()

        label_y_bottom = tk.Label(frame_adaptive, text='y_bottom:', font=self.font)
        label_y_bottom.pack()
        self.text_y_bottom = tk.Entry(frame_adaptive, font=self.font)
        self.text_y_bottom.pack()

        label_x_left = tk.Label(frame_adaptive, text='x_left:', font=self.font)
        label_x_left.pack()
        self.text_x_left = tk.Entry(frame_adaptive, font=self.font)
        self.text_x_left.pack()

        label_x_right = tk.Label(frame_adaptive, text='x_right:', font=self.font)
        label_x_right.pack()
        self.text_x_right = tk.Entry(frame_adaptive, font=self.font)
        self.text_x_right.pack()

    def start_and_status(self):
        self.label_segment = tk.Label(self.frame_right_3, text='segment width: [X]', font=self.font)
        self.label_segment.pack()
        self.progressbar = ttk.Progressbar(self.frame_right_3, length=300, cursor='watch')
        self.progressbar.pack()
        self.label_status = tk.Label(self.frame_right_3, text='Status: waiting...', font=self.font)
        self.label_status.pack()

        bt_process = tk.Button(self.frame_right_3, text='process', command=self.process, font=self.font)
        bt_process.pack()

    """
        define the callback and processing function
    """

    def display_frames(self, idx):
        self.vc.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        _, frame_1 = self.vc.read()
        _, frame_2 = self.vc.read()
        frame_overlay = (frame_1 // 2 + frame_2 // 2)
        frame_overlay = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
        ax = self.fig.add_subplot(111)
        ax.imshow(frame_overlay)
        self.canvas.draw()

    def optical_flow(self, frame_1, frame_2):
        y_top = int(self.text_y_top.get())
        y_bottom = int(self.text_y_bottom.get())
        x_left = int(self.text_x_left.get())
        x_right = int(self.text_x_right.get())

        feature_params = dict(maxCorners=20,
                              qualityLevel=0.3,
                              minDistance=3,
                              blockSize=7)
        lk_params = dict(winSize=(15, 15),
                         maxLevel=5,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

        mask = np.zeros((self.frame_height, self.frame_width), dtype='uint8')
        mask[y_top:y_bottom, x_left:x_right] = 1

        p0 = cv2.goodFeaturesToTrack(frame_1_gray, mask=mask, **feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame_1_gray, frame_2_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        moving_distance = [int(good_new[i, 0] - good_old[i, 0]) for i in range(len(good_new)) if
                           abs(good_new[i, 1] - good_old[i, 1]) < 2]
        width = max(moving_distance, default=None, key=lambda v: moving_distance.count(v))

        return width

    def open_video(self):
        self.fig.clear()

        self.video_path.set(filedialog.askopenfilename(title='choose a video'))
        self.save_path.set(os.path.dirname(self.video_path.get()))
        self.vc = cv2.VideoCapture(self.video_path.get())
        fps = self.vc.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.vc.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.label_video_attr['text'] = '... ......... ...\n' \
                                        '... fps: {:.2f} ...\n' \
                                        '... total_frames: {} ...\n' \
                                        '... resolution: {}x{} ...\n' \
                                        '... ......... ...'.format(fps,
                                                                   self.total_frames,
                                                                   self.frame_width, self.frame_height)
        self.display_frames(0)

    def select_save_dir(self):
        self.save_path.set(filedialog.askdirectory())

    def process(self):
        position = int(self.text_position.get())

        # adaptive width
        if self.v_adaptive_control.get():
            adaptive_start = int(self.text_adaptive_start.get())
            adaptive_sensitivity = int(self.text_adaptive_sensitivity.get())
            adaptive_length = self.total_frames - adaptive_start

            img_length = 0
            width_list = []
            width_adjust_position = []

            self.progressbar['maximum'] = adaptive_sensitivity
            self.label_status['text'] = 'Status: calculating movement...'
            try:
                for i in range(adaptive_sensitivity):
                    self.progressbar['value'] = i + 1
                    self.master.update()
                    self.vc.set(cv2.CAP_PROP_POS_FRAMES,
                                adaptive_start + i * ((adaptive_length - 2) // adaptive_sensitivity))
                    rval1, frame_1 = self.vc.read()
                    rval2, frame_2 = self.vc.read()
                    rval3, frame_3 = self.vc.read()
                    width_list.append((self.optical_flow(frame_1, frame_2) + self.optical_flow(frame_2, frame_3)) // 2)
                    if i == 0:
                        img_length = width_list[0] * (adaptive_start + (adaptive_length // adaptive_sensitivity))
                        width_adjust_position.append(0)
                    else:
                        img_length += width_list[i] * (adaptive_length // adaptive_sensitivity)
                        width_adjust_position.append(adaptive_start + i * (adaptive_length // adaptive_sensitivity))

                img_length += width_list[-1] * (
                        adaptive_length - (adaptive_length // adaptive_sensitivity) * adaptive_sensitivity)
            except Exception as result:
                self.label_status['text'] = f"Error: {result}"
                return

            img = np.empty((self.frame_height, abs(img_length), 3), dtype='uint8')
            self.vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
            width_list = np.array(width_list)
            width_adjust_position = np.array(width_adjust_position)

            self.label_segment['text'] = f'segment width: {width_list}'

            self.progressbar['maximum'] = self.total_frames
            self.label_status['text'] = 'Status: merging image...'

            try:
                img = self.merge_video_slides(img, width_list[0] > 0, img_length, self.total_frames,
                                              width_list, width_adjust_position, position)
            except Exception as result:
                self.label_status['text'] = f"Error: {result}"
                return

        # manual width
        else:
            self.vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
            width = float(self.text_width.get())
            img = np.empty((self.frame_height, int(self.total_frames * width), 3), dtype='uint8')

            self.progressbar['maximum'] = self.total_frames
            self.label_status['text'] = 'Status: merging image...'
            try:
                img = self.merge_video_slides(img, self.v_left_right.get(), int(self.total_frames * width),
                                              self.total_frames, np.array([math.ceil(width)]), np.array([0]), position)
            except Exception as result:
                self.label_status['text'] = f"Error: {result}"
                return

        self.save_img(img)
        return

    # todo: multiprocessing it
    def merge_video_slides(self,
                           img,
                           direction,
                           img_length,
                           total_frames,
                           width_list,
                           width_adjust_position,
                           clip_position):
        """
        key part of the video scan, to clip and paste frames into one image

        :param img: the image buffer
        :param direction: the object moving from 'left to right' = 1
        :param img_length: length of the final picture
        :param total_frames: total amount of slices, amount of frames
        :param width_list: the width of slices if width is flexible
        :param width_adjust_position: the position where width changing
        :param clip_position: the position of slices on frames
        :return: the final merged long image in BGR format, type: numpy.ndarray
        """
        vc = FileVideoStream(self.video_path.get()).start()
        pixel_start = img_length * (2 ** direction - 1)
        for i in range(total_frames):
            frame = vc.read()
            width = abs(width_list[((width_adjust_position - i) <= 0).sum() - 1])
            if direction:
                pixel_start -= width
                pixel_end = pixel_start + width
                img[:, pixel_start:pixel_end, :] = frame[:, clip_position:clip_position + width, :]
            else:
                pixel_end = pixel_start + width
                img[:, pixel_start:pixel_end, :] = frame[:, clip_position:clip_position + width, :]
                pixel_start += width
            self.progressbar['value'] = i + 1
            self.master.update()
        return img

    def save_img(self, img):
        split_width = int(self.text_split_width.get())
        self.label_status['text'] = 'Status: saving image...'
        self.master.update()
        save_dir = os.path.join(self.save_path.get(), os.path.split(self.video_path.get())[-1].split('.')[-2])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        flag = 0
        i = 0
        for i in range(img.shape[1] // split_width):
            split_start = i * split_width
            split_end = split_start + split_width
            cv2.imencode('.jpg', img[:, split_start:split_end, :])[1].tofile(
                '{}/{}_{}.jpg'.format(save_dir,
                                      os.path.split(self.video_path.get())[-1].split('.')[-2],
                                      i))
            flag = 1
        if not flag or img.shape[1] % split_width:
            i += flag
            split_start = i * split_width
            cv2.imencode('.jpg', img[:, split_start:, :])[1].tofile(
                '{}/{}_{}.jpg'.format(save_dir,
                                      os.path.split(self.video_path.get())[-1].split('.')[-2],
                                      i))

        self.label_status['text'] = 'Status: Done!'
        self.master.update()


if __name__ == '__main__':
    root = tk.Tk()
    root.title('scanning video')
    root.geometry('1024x800')
    app = Application(master=root)
    app.mainloop()
