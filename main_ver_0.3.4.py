import gc
import os
import cv2
import math
import time
from threading import Thread
from queue import Queue
import threading
import queue
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
import tkinter.font as tf
from tkinter import ttk
from tkinter import filedialog
import load_window


class FileVideoStream:
    """
    https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
    """
    def __init__(self, path, queue_size=64):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
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
                # queue is full, do not block
                # continue

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

    def running(self):
        # indicate that the thread is still running
        return not self.stopped


class VideoCaptureThread(threading.Thread):
    def __init__(self, video_file):
        threading.Thread.__init__(self)
        self.vc = cv2.VideoCapture(video_file)
        self.idx_queue = Queue(maxsize=1)
        self.current_frame_queue = Queue(maxsize=1)
        self.is_running = True

    def run(self):
        while self.is_running:
            try:
                frame_num = self.idx_queue.get(timeout=1)
                self.vc.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                _, frame_1 = self.vc.read()
                _, frame_2 = self.vc.read()
                frame_overlay = (frame_1 // 2 + frame_2 // 2)
                self.current_frame_queue.queue.clear()
                self.current_frame_queue.put_nowait(frame_overlay)
            except queue.Empty:
                pass

    def stop(self):
        self.is_running = False


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.process_flag = True
        self.master = master
        self.font = tf.Font(size=12)
        self.left_canvas()
        self.right_panel()
        self.video_args_text()
        self.pos_split_args_ctrl()
        self.video_clip_interval_ctrl()
        self.start_and_status()
        self.master.protocol("WM_DELETE_WINDOW", self.on_exit)

    """
        set the panels and widgets on GUI
    """

    def left_canvas(self):
        self.frame_left = tk.Frame(self.master)
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        self.fig = Figure(figsize=(8, 4), dpi=72)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_left)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        toolbar = NavigationToolbar2Tk(self.canvas, self.frame_left)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        self.scrollbar_display = tk.Scale(self.frame_left, orient=tk.HORIZONTAL, from_=0, to=100, resolution=1)
        self.scrollbar_display.pack(fill=tk.X)

    def right_panel(self):
        frame_right = tk.Frame(self.master)
        frame_right.pack(side=tk.LEFT, padx=10, expand=tk.YES)
        self.frame_right_1 = tk.Frame(frame_right)
        self.frame_right_1.pack()
        self.frame_right_2 = tk.Frame(frame_right)
        self.frame_right_2.pack()
        self.frame_right_seg = tk.Frame(frame_right)
        self.frame_right_seg.pack()
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

        label_split_width = tk.Label(self.frame_right_1, text='images saving width:', font=self.font)
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
        frame_segment_width = tk.Frame(self.frame_right_seg)
        frame_segment_width.pack(side=tk.LEFT, padx=5, expand=tk.YES)
        label_segment_width = tk.Label(frame_segment_width, text='segment width', font=self.font)
        label_segment_width.pack()
        self.text_segment_width = tk.Text(frame_segment_width, font=self.font, width=15, height=5, bd=2)
        self.text_segment_width.pack()

        frame_segment_pos = tk.Frame(self.frame_right_seg)
        frame_segment_pos.pack(side=tk.RIGHT, padx=5, expand=tk.YES)
        label_segment_pos = tk.Label(frame_segment_pos, text='position (frame)', font=self.font)
        label_segment_pos.pack()
        self.text_segment_pos = tk.Text(frame_segment_pos, font=self.font, width=15, height=5, bd=2)
        self.text_segment_pos.pack()

        self.progressbar = ttk.Progressbar(self.frame_right_3, length=300, cursor='watch')
        self.progressbar.pack()
        self.label_status = tk.Label(self.frame_right_3, text='Status: waiting...', font=self.font)
        self.label_status.pack()

        frame_bt_process = tk.Frame(self.frame_right_3)
        frame_bt_process.pack(side=tk.LEFT, padx=5, expand=tk.YES)
        bt_process = tk.Button(frame_bt_process, text='process',
                               command=self.process, font=self.font)
        bt_process.pack()

        frame_bt_process = tk.Frame(self.frame_right_3)
        frame_bt_process.pack(side=tk.LEFT, padx=5, expand=tk.YES)
        bt_calculate_mov = tk.Button(frame_bt_process, text='calculate move',
                                     command=self.calculate_split_wp, font=self.font)
        bt_calculate_mov.pack()

        frame_bt_stop = tk.Frame(self.frame_right_3)
        frame_bt_stop.pack(side=tk.RIGHT, padx=5, expand=tk.YES)
        bt_stop = tk.Button(frame_bt_stop, text='stop',
                            command=self.stop_process, font=self.font)
        bt_stop.pack()

    """
        define the callback and processing function
    """

    def show_on_canvas(self, frame):
        self.fig.clear()
        self.fig.add_subplot(111).imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.canvas.draw()

    def update_frame(self, frame_num):
        self.VCT.idx_queue.queue.clear()
        self.VCT.idx_queue.put_nowait(int(frame_num))
        self.show_on_canvas(self.VCT.current_frame_queue.get(timeout=1))

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
        self.scrollbar_display.pack_forget()
        self.scrollbar_display = tk.Scale(self.frame_left, orient=tk.HORIZONTAL, from_=0, to=self.total_frames-2,
                                          resolution=1, command=self.update_frame)
        self.scrollbar_display.pack(fill=tk.X)

        try:
            self.VCT.stop()
            self.VCT.join()
        except:
            pass
        self.VCT = VideoCaptureThread(self.video_path.get())
        self.VCT.start()
        self.update_frame(0)

    def select_save_dir(self):
        self.save_path.set(filedialog.askdirectory())

    def stop_process(self):
        self.video_stream.stop()
        self.progressbar['value'] = 0
        self.label_status['text'] = 'Stopped'
        self.master.update()
        self.process_flag = False

    def calculate_split_wp(self):
        self.text_segment_pos.delete('1.0', 'end')
        self.text_segment_width.delete('1.0', 'end')

        adaptive_start = int(self.text_adaptive_start.get())
        adaptive_sensitivity = int(self.text_adaptive_sensitivity.get())
        adaptive_length = self.total_frames - adaptive_start
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
                    width_adjust_position.append(0)
                else:
                    width_adjust_position.append(adaptive_start + i * (adaptive_length // adaptive_sensitivity))

            width_list_str = '\n'.join(str(num) for num in width_list)
            self.text_segment_width.insert('1.0', width_list_str)
            width_adjust_position_str = '\n'.join(str(num) for num in width_adjust_position)
            self.text_segment_pos.insert('1.0', width_adjust_position_str)

            self.label_status['text'] = '↑ Splitting info (editable) ↑'
            self.master.update()

        except Exception as result:
            self.label_status['text'] = f"Error: {result}"
            return

    def calculate_img_len(self, width_list, width_adjust_position):
        """
        The number in width_adjust_position is the absolute position in video
        the first '0' represents the adaptive_start position
        """
        img_length = 0
        for idx in range(1, len(width_adjust_position)):
            width = width_list[idx-1]
            num_frames = width_adjust_position[idx] - width_adjust_position[idx-1]
            img_length += width * num_frames
        img_length += width_list[idx] * (self.total_frames - width_adjust_position[idx])

        return img_length

    def process(self):
        self.FVS = FileVideoStream(self.video_path.get())
        self.process_flag = True
        detect_position = int(self.text_position.get())

        # adaptive width
        if self.v_adaptive_control.get():
            width_list = self.text_segment_width.get('1.0', 'end')
            width_list = [int(num.rstrip()) for num in width_list.split('\n') if num]
            width_adjust_position = self.text_segment_pos.get('1.0', 'end')
            width_adjust_position = [int(num.rstrip()) for num in width_adjust_position.split('\n') if num]
            width_list = np.array(width_list)
            width_adjust_position = np.array(width_adjust_position)

            img_length = self.calculate_img_len(width_list, width_adjust_position)

            img = np.empty((self.frame_height, abs(img_length), 3), dtype='uint8')
            self.vc.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.progressbar['maximum'] = self.total_frames
            self.label_status['text'] = 'Status: merging image...'

            try:
                img = self.merge_video_slides(img, width_list[0] > 0, img_length, self.total_frames,
                                              width_list, width_adjust_position, detect_position)
            except Exception as result:
                self.label_status['text'] = f"Error: {result}"
                return

        # manual width
        else:
            self.vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
            width = float(self.text_width.get())
            img = np.zeros((self.frame_height, int(self.total_frames * width), 3), dtype='uint8')

            self.progressbar['maximum'] = self.total_frames
            self.label_status['text'] = 'Status: merging image...'
            try:
                img = self.merge_video_slides(img, self.v_left_right.get(), int(self.total_frames * width),
                                              self.total_frames,
                                              np.array([math.ceil(width)]), np.array([0]), detect_position)
            except Exception as result:
                self.label_status['text'] = f"Error: {result}"
                return
        if self.process_flag:
            self.save_img(img)
        return

    # todo: speed up it
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
        self.video_stream = self.FVS.start()
        pixel_start = img_length * (2 ** direction - 1)
        scrollbar_flag = self.scrollbar_display.get()
        for i in range(total_frames):
            if self.process_flag:
                frame = self.video_stream.read()
                width = abs(width_list[((width_adjust_position - i) <= 0).sum() - 1])
                if direction:
                    pixel_start -= width
                    pixel_end = pixel_start + width
                    img[:, pixel_start:pixel_end, :] = frame[:, clip_position:clip_position + width, :]
                else:
                    pixel_end = pixel_start + width
                    img[:, pixel_start:pixel_end, :] = frame[:, clip_position:clip_position + width, :]
                    pixel_start += width
                if i % 500 == 0:
                    self.scrollbar_display.set(i)
                self.progressbar['value'] = i + 1
                self.master.update()
            else:
                break
        self.scrollbar_display.set(scrollbar_flag)
        self.update_frame(scrollbar_flag)
        return img

    def save_img(self, img):
        split_width = int(self.text_split_width.get())
        self.label_status['text'] = 'Status: saving image...'
        self.master.update()
        save_dir = os.path.join(self.save_path.get(), os.path.split(self.video_path.get())[-1].split('.')[-2])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(img.shape[1] // split_width + 1):
            split_start = i * split_width
            split_end = split_start + split_width
            if i == img.shape[1] // split_width:
                split_end = img.shape[1]
            cv2.imencode('.jpg',
                         img[:, split_start:split_end, :],
                         [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tofile('{}/{}_{}.jpg'.format(save_dir,
                                      os.path.split(self.video_path.get())[-1].split('.')[-2], i))

        self.label_status['text'] = 'Status: Done!'
        self.master.update()

    def on_exit(self):
        try:
            self.process_flag = False
            self.VCT.stop()
            self.VCT.join()
            self.FVS.stop()
        except:
            pass
        self.master.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    root.title('scanning video')
    root.geometry('1024x800')
    app = Application(master=root)
    app.mainloop()
