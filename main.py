#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
import tkinter.font as tf
from tkinter import *
from tkinter import ttk
from tkinter import filedialog


def display_frames(idx):
    global vc, canvas, fig
    idx = int(idx)
    vc.set(cv2.CAP_PROP_POS_FRAMES, idx)
    rval, frame_1 = vc.read()
    rval, frame_2 = vc.read()
    frame_overlay = ((frame_1.astype(np.int) + frame_2.astype(np.int)) * 0.5).astype(np.uint8)
    frame_overlay = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(111)
    ax.imshow(frame_overlay)
    canvas.draw()


def open_video():
    global fig, vc, total_frames, frame_height
    fig.clear()

    video_path.set(filedialog.askopenfilename(title='choose a video'))
    vc = cv2.VideoCapture(video_path.get())
    fps = vc.get(cv2.CAP_PROP_FPS)
    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    label_video_attr['text'] = '... ......... ...\n' \
                               '... fps: {} ...\n' \
                               '... total_frames: {} ...\n' \
                               '... resolution: {}x{} ...\n' \
                               '... ......... ...'.format(fps, total_frames, frame_width, frame_height)
    display_frames(0)


def save_img():
    save_path.set(filedialog.askdirectory())


def process():
    global vc, total_frames, frame_height

    label_status['text'] = 'Status: waiting...'
    position = int(text_position.get())
    width = float(text_width.get())
    split_width = int(text_split_width.get())

    print('pos:{} width:{} sp_width:{}'.format(position, width, split_width))

    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)

    img = np.empty((frame_height, int(total_frames * width), 3), dtype='uint8')

    progressbar['maximum'] = total_frames
    for i in range(total_frames):
        rval, frame = vc.read()
        if not rval:
            print('break')
            label_status['text'] = 'Status: *break*'
            break

        if v_left_right.get():
            pixel_start = int(total_frames * width) - int((i + 1) * width)
        else:
            pixel_start = int(i * width)
        pixel_end = pixel_start + math.ceil(width)

        img[:, pixel_start:pixel_end, :] = frame[:, position:position + math.ceil(width), :]
        progressbar['value'] = i + 1
        window.update()

    save_dir = os.path.join(save_path.get(), os.path.split(video_path.get())[-1].split('.')[-2])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    flag = 0
    i = 0
    for i in range(int((total_frames * width) / split_width)):
        split_start = i * split_width
        split_end = split_start + split_width
        cv2.imwrite('{}/{}_{}.jpg'.format(save_dir, os.path.split(video_path.get())[-1].split('.')[-2], i),
                    img[:, split_start:split_end, :])
        flag = 1
    if not flag or int(total_frames * width) % split_width:
        i += flag
        split_start = i * split_width
        cv2.imwrite('{}/{}_{}.jpg'.format(save_dir, os.path.split(video_path.get())[-1].split('.')[-2], i),
                    img[:, split_start:, :])

    label_status['text'] = 'Status: Done!'


if __name__ == '__main__':
    window = tk.Tk()
    window.title('scanning video')
    window.geometry('1024x500')

    font = tf.Font(size=12)

    frame_left = Frame(window)
    frame_left.pack(side=LEFT, fill=BOTH, expand=YES)

    fig = Figure(figsize=(8, 4), dpi=72)
    canvas = FigureCanvasTkAgg(fig, master=frame_left)
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)
    toolbar = NavigationToolbar2Tk(canvas, frame_left)
    toolbar.update()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)

    scrollbar_display = Scale(frame_left, orient=HORIZONTAL, from_=0, to=500,
                              resolution=1, command=display_frames)
    scrollbar_display.pack(fill=X)

    ##########################################################

    frame_right = Frame(window)
    frame_right.pack(side=LEFT, padx=10, expand=YES)

    video_path = StringVar()
    video_path.set('... select a video ...')
    label_video_path = tk.Label(frame_right, textvariable=video_path, font=font)
    label_video_path.pack()
    bt_open_video = tk.Button(frame_right, text='open a video', command=open_video, font=font)
    bt_open_video.pack()

    save_path = StringVar()
    save_path.set('... select save dir ...')
    label_save_path = tk.Label(frame_right, textvariable=save_path, font=font)
    label_save_path.pack()
    bt_save_img = tk.Button(frame_right, text='save dir', command=save_img, font=font)
    bt_save_img.pack()

    label_video_attr = tk.Label(frame_right, font=font, text='... ......... ...\n'
                                                             '... info area ...\n'
                                                             '... ......... ...')
    label_video_attr.pack()

    ##########################################################

    label_position = tk.Label(frame_right, text='position:', font=font)
    label_position.pack()
    text_position = tk.Entry(frame_right, font=font)
    text_position.pack()

    label_width = tk.Label(frame_right, text='width:', font=font)
    label_width.pack()
    text_width = tk.Entry(frame_right, font=font)
    text_width.pack()

    label_split_width = tk.Label(frame_right, text='split_width:', font=font)
    label_split_width.pack()
    text_split_width = tk.Entry(frame_right, font=font)
    text_split_width.pack()

    ##########################################################

    progressbar = ttk.Progressbar(frame_right, length=300, cursor='watch')
    progressbar.pack()
    label_status = tk.Label(frame_right, text='Status: waiting...', font=font)
    label_status.pack()

    v_left_right = IntVar()
    radio_left_to_right = tk.Radiobutton(frame_right, text='left to right',
                                          variable=v_left_right, value=1, font=font)
    radio_left_to_right.pack()
    radio_right_to_left = tk.Radiobutton(frame_right, text='right to left',
                                          variable=v_left_right, value=0, font=font)
    radio_right_to_left.pack()
    bt_process = tk.Button(frame_right, text='process', command=process, font=font)
    bt_process.pack()

    window.mainloop()


'''

video_path = './DSCF1150.MOV'
position = 960
width = 10
split_width = 10000
save_path = 'C:/Users/Administrator/Desktop/new'

vc = cv2.VideoCapture(video_path)
rval = vc.isOpened()

rval, frame_1 = vc.read()
rval, frame_2 = vc.read()

frame_overlay = ((frame_1.astype(np.int) + frame_2.astype(np.int)) * 0.5).astype(np.uint8)
# cv2.imwrite('overlay.jpg', frame_overlay)

frame_overlay = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
plt.imshow(frame_overlay)
plt.show()




fps = vc.get(cv2.CAP_PROP_FPS)
print('fps: {}'.format(fps))
total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print('total frames: {}'.format(total_frames))

img = np.empty((1080, total_frames*width, 3), dtype='uint8')

for i in range(total_frames):
    rval, frame = vc.read()
    if not rval:
        print('break')
        break
    pixel_start = total_frames*width - (i+1)*width
    pixel_end = pixel_start + width
    img[:, pixel_start:pixel_end, :] = frame[:, position:position+width, :]
    
for i in range(int(total_frames*width/split_width)):
    split_start = i * split_width
    split_end = split_start + split_width
    cv2.imwrite('{}/{}_{}.jpg'.format(save_path, os.path.split(video_path)[-1].split('.')[-2], i),
            img[:, split_start:split_end, :])

i += 1
split_start = i * split_width
cv2.imwrite('{}/{}_{}.jpg'.format(save_path, os.path.split(video_path)[-1].split('.')[-2], i),
            img[:, split_start:, :])

'''
