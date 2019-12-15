from PhaseBased import modify_motion, modify_motion_mp
import utils
import sys, os
import argparse
import logging
import numpy as np
import cv2


if __name__ ==  '__main__':

    frame_path = "frames"
    try:
        os.mkdir(frame_path)
    except OSError:
        print ("Creation of the directory %s failed" % frame_path)
    else:
        print ("Successfully created the directory %s " % frame_path)

    #video to frames
    utils.video2imageFolder('crane_crop.mp4', "frames")

    dir_frames = 'frames'
    filenames = []
    filesinfo = os.scandir(dir_frames)
    filenames = [f.path for f in filesinfo if f.name.endswith(".jpg")]
    filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    frameCount = len(filenames)
    frameHeight, frameWidth, frameChannels = cv2.imread(filenames[0]).shape
    frames = np.zeros((frameCount, frameHeight, frameWidth, frameChannels),dtype=np.float32)

    for idx, file_i in enumerate(filenames):
        frames[idx] = cv2.cvtColor(cv2.imread(file_i), cv2.COLOR_BGR2LAB)



    ##!!!!!!!!!!!!!!!!!!! check the last index to the channel you are working on, L = 0, AB = 12
    L = frames[:,:,:,0]


    mL = modify_motion_mp(L,76,3,2,8,24,0.2,0.25)

    magnified_frame_path = "m_frames"
    try:
        os.mkdir(magnified_frame_path)
    except OSError:
        print ("Creation of the directory %s failed" % magnified_frame_path)
    else:
        print ("Successfully created the directory %s " % magnified_frame_path)
    #save magnified frames
    for i in range(mL.shape[0]):
        cv2.imwrite('m_frames/b{:04d}.jpg'.format(i), mL[i].clip(0,255).astype(np.uint8))

    #convert frames to video
    utils.imageFolder2mpeg("m_frames", fps=24.0)
