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


def optical_flow(frame_1, frame_2):
    global frame_height, frame_width, y_top, y_bottom, x_left, x_right
    feature_params = dict(maxCorners=20,
                          qualityLevel=0.3,
                          minDistance=3,
                          blockSize=7)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros((frame_height, frame_width), dtype='uint8')
    mask[y_top:y_bottom, x_left:x_right] = 1
    
    p0 = cv2.goodFeaturesToTrack(frame_1_gray, mask=mask, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame_1_gray, frame_2_gray, p0, None, **lk_params)
    
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    moving_distance = [int(good_new[i, 0]-good_old[i, 0]) for i in range(len(good_new)) if abs(good_new[i, 1]-good_old[i, 1]) < 2]
    width = max(moving_distance, default=None, key=lambda v: moving_distance.count(v))

    return width


def open_video():
    global fig, vc, total_frames, frame_height, frame_width
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
    global vc, total_frames, frame_height, position, y_top, y_bottom, x_left, x_right

    position = int(text_position.get())
    split_width = int(text_split_width.get())

    # adaptive width
    if v_adaptive_control.get():
        adaptive_start = int(text_adaptive_start.get())
        adaptive_sensitivity = int(text_adaptive_sensitivity.get())
        adaptive_length = total_frames - adaptive_start
        y_top = int(text_y_top.get())
        y_bottom = int(text_y_bottom.get())
        x_left = int(text_x_left.get())
        x_right = int(text_x_right.get())
        
        img_length = 0
        width_list = []
        width_adjust_position = []
        
        progressbar['maximum'] = adaptive_sensitivity
        label_status['text'] = 'Status: calculating movement...'
        for i in range(adaptive_sensitivity):
            progressbar['value'] = i + 1
            window.update()
            vc.set(cv2.CAP_PROP_POS_FRAMES, adaptive_start + i * (adaptive_length//adaptive_sensitivity))
            rval, frame_1 = vc.read()
            rval, frame_2 = vc.read()
            rval, frame_3 = vc.read()
            width_list.append((optical_flow(frame_1, frame_2)+optical_flow(frame_2, frame_3))//2)
            if i == 0:
                img_length = width_list[0] * (adaptive_start + (adaptive_length//adaptive_sensitivity))
                width_adjust_position.append(0)
            else:
                img_length += width_list[i] * (adaptive_length//adaptive_sensitivity)
                width_adjust_position.append(adaptive_start + i * (adaptive_length//adaptive_sensitivity))
        img_length += width_list[-1] * (adaptive_length - (adaptive_length//adaptive_sensitivity) * adaptive_sensitivity)
            
        img = np.empty((frame_height, abs(img_length), 3), dtype='uint8')
        vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
        width_list = np.array(width_list)
        width_adjust_position = np.array(width_adjust_position)
        
        progressbar['maximum'] = total_frames
        label_status['text'] = 'Status: merging image...'
        if width_list[0] > 0:
            pixel_start = img_length
            for i in range(total_frames):
                rval, frame = vc.read()
                if not rval:
                    label_status['text'] = 'Status: *break*'
                    break
                
                width = width_list[((width_adjust_position - i) <= 0).sum() -1]
                
                pixel_start -= width
                pixel_end = pixel_start + width
                img[:, pixel_start:pixel_end, :] = frame[:, position:position + width, :]
                
                progressbar['value'] = i + 1
                window.update()
                    
        else:
            pixel_start = 0
            for i in range(total_frames):
                rval, frame = vc.read()
                if not rval:
                    label_status['text'] = 'Status: *break*'
                    break
                width = abs(width_list[((width_adjust_position - i) <= 0).sum() -1])
                
                pixel_end = pixel_start + width
                img[:, pixel_start:pixel_end, :] = frame[:, position:position + width, :]
                pixel_start += width
                
                progressbar['value'] = i + 1
                window.update()

    # manual width
    else:
        vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
        width = float(text_width.get())
        img = np.empty((frame_height, int(total_frames * width), 3), dtype='uint8')
    
        progressbar['maximum'] = total_frames
        label_status['text'] = 'Status: merging image...'
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
    
    label_status['text'] = 'Status: saving image...'
    window.update()
    save_dir = os.path.join(save_path.get(), os.path.split(video_path.get())[-1].split('.')[-2])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    flag = 0
    i = 0
    for i in range(img.shape[1] // split_width):
        split_start = i * split_width
        split_end = split_start + split_width
        cv2.imwrite('{}/{}_{}.jpg'.format(save_dir, os.path.split(video_path.get())[-1].split('.')[-2], i),
                    img[:, split_start:split_end, :])
        flag = 1
    if not flag or img.shape[1] % split_width:
        i += flag
        split_start = i * split_width
        cv2.imwrite('{}/{}_{}.jpg'.format(save_dir, os.path.split(video_path.get())[-1].split('.')[-2], i),
                    img[:, split_start:, :])
    
    label_status['text'] = 'Status: Done!'
    window.update()


window = tk.Tk()
window.title('scanning video')
window.geometry('1024x800')

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

frame_right_1 = Frame(frame_right)
frame_right_1.pack()

video_path = StringVar()
video_path.set('... select a video ...')
label_video_path = tk.Label(frame_right_1, textvariable=video_path, font=font)
label_video_path.pack()
bt_open_video = tk.Button(frame_right_1, text='open a video', command=open_video, font=font)
bt_open_video.pack()

save_path = StringVar()
save_path.set('... select save dir ...')
label_save_path = tk.Label(frame_right_1, textvariable=save_path, font=font)
label_save_path.pack()
bt_save_img = tk.Button(frame_right_1, text='save dir', command=save_img, font=font)
bt_save_img.pack()

label_video_attr = tk.Label(frame_right_1, font=font, text='... ......... ...\n'
                                                         '... info area ...\n'
                                                         '... ......... ...')
label_video_attr.pack()

##########################################################

label_position = tk.Label(frame_right_1, text='position:', font=font)
label_position.pack()
text_position = tk.Entry(frame_right_1, font=font)
text_position.pack()

label_split_width = tk.Label(frame_right_1, text='split_width:', font=font)
label_split_width.pack()
text_split_width = tk.Entry(frame_right_1, font=font)
text_split_width.pack()

##########################################################
frame_right_2 = Frame(frame_right)
frame_right_2.pack()

v_adaptive_control = IntVar()

# manual panel
frame_manual = Frame(frame_right_2)
frame_manual.pack(side=LEFT, expand=YES)

radio_manual_control = tk.Radiobutton(frame_manual, text='manual',
                                      variable=v_adaptive_control, value=0, font=font)
radio_manual_control.pack()

label_width = tk.Label(frame_manual, text='width:', font=font)
label_width.pack()
text_width = tk.Entry(frame_manual, font=font)
text_width.pack()

v_left_right = IntVar()
radio_left_to_right = tk.Radiobutton(frame_manual, text='left to right',
                                      variable=v_left_right, value=1, font=font)
radio_left_to_right.pack()
radio_right_to_left = tk.Radiobutton(frame_manual, text='right to left',
                                      variable=v_left_right, value=0, font=font)
radio_right_to_left.pack()


# adaptive panel
frame_adaptive = Frame(frame_right_2)
frame_adaptive.pack(side=RIGHT, padx=10, expand=YES)

radio_adaptive_control = tk.Radiobutton(frame_adaptive, text='adaptive',
                                      variable=v_adaptive_control, value=1, font=font)
radio_adaptive_control.pack()

label_adaptive_start = tk.Label(frame_adaptive, text='start frame:', font=font)
label_adaptive_start.pack()
text_adaptive_start = tk.Entry(frame_adaptive, font=font)
text_adaptive_start.pack()

label_adaptive_sensitivity = tk.Label(frame_adaptive, text='sensitivity:', font=font)
label_adaptive_sensitivity.pack()
text_adaptive_sensitivity = tk.Entry(frame_adaptive, font=font)
text_adaptive_sensitivity.pack()

label_y_top = tk.Label(frame_adaptive, text='mask y_top:', font=font)
label_y_top.pack()
text_y_top = tk.Entry(frame_adaptive, font=font)
text_y_top.pack()

label_y_bottom = tk.Label(frame_adaptive, text='y_bottom:', font=font)
label_y_bottom.pack()
text_y_bottom = tk.Entry(frame_adaptive, font=font)
text_y_bottom.pack()

label_x_left = tk.Label(frame_adaptive, text='x_left:', font=font)
label_x_left.pack()
text_x_left = tk.Entry(frame_adaptive, font=font)
text_x_left.pack()

label_x_right = tk.Label(frame_adaptive, text='x_right:', font=font)
label_x_right.pack()
text_x_right = tk.Entry(frame_adaptive, font=font)
text_x_right.pack()

##########################################################    
frame_right_3 = Frame(frame_right)
frame_right_3.pack()

progressbar = ttk.Progressbar(frame_right_3, length=300, cursor='watch')
progressbar.pack()
label_status = tk.Label(frame_right_3, text='Status: waiting...', font=font)
label_status.pack()

bt_process = tk.Button(frame_right_3, text='process', command=process, font=font)
bt_process.pack()


window.mainloop()




"""

# https://zhuanlan.zhihu.com/p/105998058
import numpy as np
import cv2
import sys

cap = cv2.VideoCapture("cover.mov")
feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0,255,(100,3))
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)#选取好的特征点，返回特征点列表
mask = np.zeros_like(old_frame)

for i in range(80):
    ret,frame = cap.read()
    if frame is None: break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)#计算新的一副图像中相应的特征点额位置
    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel() #ravel()函数用于降维并且不产生副本
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 3)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    # cv2.imshow('frame',img)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)


cv2.imwrite('mask.jpg', mask)
cv2.imwrite('frame.jpg', frame)

cv2.destroyAllWindows()
cap.release()

"""

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
"""

video_path = './DSCF2438.mov'
position = 180
width = 10
split_width = 10000
save_path = './'

vc = cv2.VideoCapture(video_path)
rval = vc.isOpened()
vc.set(cv2.CAP_PROP_POS_FRAMES, 170)
rval, frame_1 = vc.read()

rval, frame_2 = vc.read()

frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

frame_overlay = ((frame_1.astype(np.int) + frame_2.astype(np.int)) * 0.5).astype(np.uint8)
# cv2.imwrite('overlay.jpg', frame_overlay)
# frame_overlay = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
# plt.imshow(frame_overlay)
# plt.show()

fps = vc.get(cv2.CAP_PROP_FPS)
total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('fps: {}  total_frames: {}  resolution: {}x{}'.format(fps, total_frames, frame_width, frame_height))


# optical flow
feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=3,
                      blockSize=5)
lk_params = dict(winSize=(15,15),
                  maxLevel=5,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

mask = np.zeros((frame_height, frame_width), dtype='uint8')
mask[frame_height//2:frame_height//2+600, position-300:position+300] = 1
plt.imshow(mask)
plt.show()

p0 = cv2.goodFeaturesToTrack(frame_1_gray, mask=None, **feature_params)
p1, st, err = cv2.calcOpticalFlowPyrLK(frame_1_gray, frame_2_gray, p0, None, **lk_params)

good_new = p1[st==1]
good_old = p0[st==1]

line = np.zeros_like(frame_1)
frame = frame_overlay
for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    line = cv2.line(line, (a,b), (c,d), [0,255,255], 1)
    frame = cv2.circle(frame, (a,b), 3, [0,0,255], -1)
    frame = cv2.circle(frame, (c,d), 3, [0,255,0], -1)
img = cv2.add(frame, line)
cv2.imwrite('optical_flow.jpg', img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

moving_distance = [int(good_new[i, 0]-good_old[i, 0]) for i in range(len(good_new)) if abs(good_new[i, 1]-good_old[i, 1]) < 2]
width = max(moving_distance, default=None, key=lambda v: moving_distance.count(v))


#####################################################################



#####################################################################
# def process():
adaptive_start = 180
adaptive_length = total_frames - adaptive_start
adaptive_sensitivity = 7

y_top = 750
y_bottom = 1450
x_left = 100
x_right = 600


img_length = 0
width_list = []
width_adjust_position = []
for i in range(adaptive_sensitivity):
    print(adaptive_start + i * (adaptive_length//adaptive_sensitivity))
    vc.set(cv2.CAP_PROP_POS_FRAMES, adaptive_start + i * (adaptive_length//adaptive_sensitivity))
    rval, frame_1 = vc.read()
    rval, frame_2 = vc.read()
    rval, frame_3 = vc.read()
    width_list.append((optical_flow(frame_1, frame_2)+optical_flow(frame_2, frame_3))//2)
    if i == 0:
        img_length = width_list[0] * (adaptive_start + (adaptive_length//adaptive_sensitivity))
        width_adjust_position.append(0)
    else:
        img_length += width_list[i] * (adaptive_length//adaptive_sensitivity)
        width_adjust_position.append(adaptive_start + i * (adaptive_length//adaptive_sensitivity))
img_length += width_list[-1] * (adaptive_length - (adaptive_length//adaptive_sensitivity) * adaptive_sensitivity)
        
    
img = np.empty((frame_height, abs(img_length), 3), dtype='uint8')
vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
width_list = np.array(width_list)
width_adjust_position = np.array(width_adjust_position)

if width_list[0] > 0:
    pixel_start = img_length
    for i in range(total_frames):
        rval, frame = vc.read()
        if not rval:
            print('break')
            break
        
        width = width_list[((width_adjust_position - i) <= 0).sum() -1]
        
        pixel_start -= width
        pixel_end = pixel_start + width
        img[:, pixel_start:pixel_end, :] = frame[:, position:position + width, :]
        
        if i % 100 == 0:
            print('{}/{} - {}'.format(i, total_frames, width))
            
else:
    pixel_start = 0
    for i in range(total_frames):
        rval, frame = vc.read()
        if not rval:
            print('break')
            break
        width = abs(width_list[((width_adjust_position - i) <= 0).sum() -1])
        
        pixel_end = pixel_start + width
        img[:, pixel_start:pixel_end, :] = frame[:, position:position + width, :]
        pixel_start += width
        
        if i % 100 == 0:
            print('{}/{} - {}'.format(i, total_frames, width))


'''
# too slow!!

vc.set(cv2.CAP_PROP_POS_FRAMES, adaptive_start)

rval, frame_1 = vc.read()
rval, frame_2 = vc.read()
width = optical_flow(frame_1, frame_2)



vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
rval, frame = vc.read()
img = frame[:, position:position+abs(width), :]

for i in range(1, total_frames):
    rval, frame = vc.read()
    if not rval:
        print('break')
        break

    if width > 0:  # left --> right
        img = np.concatenate((frame[:, position:position+abs(width), :], img), axis=1)
    else:  # right --> left
        img = np.concatenate((img, frame[:, position:position+abs(width), :]), axis=1)
    if i % 100 == 0:
        print('{}/{}'.format(i, total_frames))
    
    if i > adaptive_start and (i - adaptive_start) % (adaptive_length//adaptive_sensitivity) == 0:
        vc.set(cv2.CAP_PROP_POS_FRAMES, i-1)
        rval, frame_1 = vc.read()
        rval, frame_2 = vc.read()
        width = optical_flow(frame_1, frame_2)
        print(width)
'''
    

save_dir = os.path.join('./', os.path.split(video_path)[-1].split('.')[-2])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
flag = 0
i = 0
for i in range(img.shape[1] // split_width):
    split_start = i * split_width
    split_end = split_start + split_width
    cv2.imwrite('{}/{}_{}.jpg'.format(save_dir, os.path.split(video_path)[-1].split('.')[-2], i),
                img[:, split_start:split_end, :])
    flag = 1
if not flag or img.shape[1] % split_width:
    i += flag
    split_start = i * split_width
    cv2.imwrite('{}/{}_{}.jpg'.format(save_dir, os.path.split(video_path)[-1].split('.')[-2], i),
                img[:, split_start:, :])
    
# cv2.imwrite('./test.jpg', img[:, 0:20000, :])

# img = np.empty((1080, total_frames*width, 3), dtype='uint8')

# for i in range(total_frames):
#     rval, frame = vc.read()
#     if not rval:
#         print('break')
#         break
#     pixel_start = total_frames*width - (i+1)*width
#     pixel_end = pixel_start + width
#     img[:, pixel_start:pixel_end, :] = frame[:, position:position+width, :]
    
# for i in range(int(total_frames*width/split_width)):
#     split_start = i * split_width
#     split_end = split_start + split_width
#     cv2.imwrite('{}/{}_{}.jpg'.format(save_path, os.path.split(video_path)[-1].split('.')[-2], i),
#             img[:, split_start:split_end, :])

# i += 1
# split_start = i * split_width
# cv2.imwrite('{}/{}_{}.jpg'.format(save_path, os.path.split(video_path)[-1].split('.')[-2], i),
#             img[:, split_start:, :])
"""
