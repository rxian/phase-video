from PhaseBased import modify_motion
import utils
import sys, os
import argparse
import logging
import numpy as np
import cv2


SCRIPT_NAME = os.path.basename(__file__)

LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))

'''
$python  results.py -a 100 -d 3 -n 2 -k 8 -s 24 -l 0.2 -f 0.5 -i "crane_crop.mp4"
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PhaseBased')
    parser.add_argument('--alpha', '-a', default=100, type=int, help='alpha')
    parser.add_argument('--D', '-d', default=3, type=int,help='Depth')
    parser.add_argument('--N', '-n', default=2, type=int,help='number of filters')
    parser.add_argument('--K', '-k', default=8, type=int,help='Orientation. Integer')
    parser.add_argument('--fs', '-s', default=24, type = int, help='fs')
    parser.add_argument('--fl', '-l', default=0.2, type = float,help='fl')
    parser.add_argument('--fh', '-f', default=0.5, type = float,help='fh')
    parser.add_argument('--input_file', '-i', default='crane_crop.mp4',help='input video')
    args = parser.parse_args()

    '''
    frame_path = "frames"
    try:
        os.mkdir(frame_path)
    except OSError:
        print ("Creation of the directory %s failed" % frame_path)
    else:
        print ("Successfully created the directory %s " % frame_path)
    #video to frames
    utils.video2imageFolder(args.input_file, "frames")
    '''
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

    print("generate motion magnified frames")
    ret = np.zeros(frames.shape)
    for i in range(3):
        print("processing channel %d" %i)
        ret[:,:,:,i] = modify_motion(frames[:,:,:,i],args.alpha,args.D,args.N,args.K,args.fs,args.fl,args.fh)

    print("modify_motion completed")
    magnified_frame_path = "m_frames"
    try:
        os.mkdir(magnified_frame_path)
    except OSError:
        print ("Creation of the directory %s failed" % magnified_frame_path)
    else:
        print ("Successfully created the directory %s " % magnified_frame_path)
    #save magnified frames
    for i in range(ret.shape[0]):
        im_out = ret[i,:,:,:].astype(np.uint8)
        cv2.imwrite('m_frames/b{:04d}.jpg'.format(i), cv2.cvtColor(im_out, cv2.COLOR_LAB2RGB))
    #convert frames to video
    utils.imageFolder2mpeg("m_frames", fps=30.0)
