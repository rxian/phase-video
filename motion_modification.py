import numpy as np
import scipy
from temporal_modification import perform_temporal_modification
from ComplexSteerablePyramid import pyr2im, im2pyr

def modify_motion(frames, frame_rate, alpha, D, N, K, F_l, F_h):
    ''' Modify motion by a factor of alpha using firwin as temporal filter
    Input:
        - frames: a list of frames converted from the video to be modified
        - frame_rate: frame rate of the video
        - alpha: the factor by which motion in the video should be modified
        - D: depth of pyramid
        - N: number of filters per octave to be used for constructing pyramid
        - K: number of orientations
        - pyr_filter: the filter to be used for constructing pyramid
        - F_l: low-pass frequency for temporal filter
        - F_h: high_pass frequency for temporal filter
        - [Deprecated] temporal_filter: the filter to be used for temporal filtering
    Output:
        - a list of modified frames that can be converted to a video
    '''
    T = len(frames)

    # obtain steerable pyramid of shape (T, D, N, K)
    pyr = [[[[[] for i in range(K)] for i in range(N)] for i in range(D)] for t in range(T)]
    residual_h, residual_l = [], []
    for t, frame in enumerate(frames):
        p, rh, rl = im2pyr(frame, D, N, K)
        
        pyr[t] = p
        residual_h.append(rh)
        residual_l.append(rl)

    # modify motion
    modified_pyr = [[[[[] for i in range(K)] for i in range(N)] for i in range(D)] for t in range(T)]
    temporal_filter = scipy.fftpack.fft(scipy.fftpack.ifftshift(scipy.signal.firwin(T,[F_l,F_h],fs=frame_rate,pass_zero=False)))
    for d in range(D):
        for n in range(N):
            for k in range(K):
                og = [fr_pyr[d][n][k] for fr_pyr in pyr]
                modified = perform_temporal_modification(og, alpha, temporal_filter)  # modified_fr_pyr of shape T
                
                for t in range(T):
                    modified_pyr[t][d][n][k] = modified[t]

    # collapse pyramid
    modified_frames = []
    for t in range(len(pyr)):
        modified_frames[t] = pyr2im(pyr[t], residual_h[t], residual_l[t])

    return modified_frames