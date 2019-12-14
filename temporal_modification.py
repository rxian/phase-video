import scipy
from scipy import fftpack, signal
import matplotlib.pyplot as plt
import numpy as np

def perform_temporal_modification(frames, alpha, temporal_filter):
    ''' Perform temporal modification of a list of frames
    Input:
        - frames: a list of 2-D numpy array
        - alpha: the factor by which motion in the video should be modified
        - temporal_filter: the filter to be used for temporal filtering
    Output: 
        -  a list of 2-D numpy array
    '''
    delta_phi = temporal_filtering(np.angle(frames), temporal_filter)
   
    J = []
    for t in range(delta_phi.shape[0]):
        J.append(modify_motion(alpha, delta_phi[t], frames[t]))
    return J

def temporal_filtering(I, F):
    '''
    Output: a 3-D numpy array of shape (T, h, w)
    '''
    _, h, w = I.shape
    J = np.empty(shape=I.shape, dtype=np.complex)
    for i in range(w*h):
        row, col = np.unravel_index(i, (h, w))
        x = I[:, row, col]
        x_dft = fftpack.fft(x)        
        y_dft = x_dft * F
        J[:, row, col] = fftpack.ifft(y_dft)
    return J

def modify_motion(alpha, delta_phi, I):
    '''
    Output: a 2-D array of shape (h, w)
    '''
    return I * np.exp((alpha - 1) * np.complex(0, 1) * delta_phi)