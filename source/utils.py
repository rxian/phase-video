'''
Utility I/O functions.  Based on UIUC CS445 course material.
'''

import numpy as np
import cv2
import os

def video2numpy(path):
    '''
    Read a video and store its frames in an uint8 numpy.ndarray
    with shape (length, height, width, BGR-channels).
    '''
    cap = cv2.VideoCapture()
    cap.open(path)

    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fs = float(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    for i in range(T):
        ret, frame = cap.read()
        if ret == 0:
            print("Failed to get frame %d" %(i))
            continue
        frames.append(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i+1)

    cap.release()

    return np.array(frames), fs

def numpy2video(path, frames, fs=30.0):
    '''
    Inverse operation of video2numpy.
    '''
    codec = cv2.VideoWriter_fourcc(*'MPG1')
    writer = cv2.VideoWriter(path, codec, fs, frames.shape[1:3])

    T = len(frames)
    for t in range(T):
        writer.write(frames[t])

    writer.release()

def numpy2folder(frames,path):
    '''
    TODO: save frames as images to a folder
    '''
    pass

def folder2numpy(path):
    '''
    TODO: load frames from sorted images in a folder.
    '''
    pass
