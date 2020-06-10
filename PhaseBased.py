import numpy as np
import scipy.fftpack, scipy.signal
from ComplexSteerablePyramid import pyr2im, im2pyr
from tqdm import tqdm

def modify_motion(frames,alpha,D,N,K,F,verbose=True):
    '''
    Perform phased-based motion processing.

    @type  frames: numpy.ndarray of shape (T,H,W)
    @param frames: Frames of a video sequence of length T.
    @type  alpha: number
    @param alpha: Amount of motion amplification/attenuation.
    @type  D: integer
    @param D: Depth of pyramid (number of octaves).
    @type  N: integer
    @param N: Number of suboctaves per octave.
    @type  K: integer
    @param K: Number of pyramid orientations.
    @type  F: numpy.ndarray of length (L,)
    @param F: Temporal filter in fourier coefficients.
    @type  verbose: boolean
    @param verbose: Print progress bar.
    @rtype:   numpy.ndarray of shape (T,H,W)
    @return:  Edited frames.
    '''
    T = len(frames)
    pad = len(F) - T
    Ps, Rhs, Rls = [], [], []

    ## Transform frames to pyramid representation
    print("Converting original frames to pyramid")
    Ps, Rhs, Rls = im2pyr(frames,D,N,K,verbose=verbose)

    ## Motion editting
    print("Modifying motion")
    if verbose: pbar = tqdm(total=D*N*K)
    for d in range(D):
        for n in range(N):
            for k in range(K):
                P_frames = np.pad(Ps[d][n][k],((0,pad),(0,0),(0,0)),mode='edge')
                P_frames = np.moveaxis(P_frames,0,-1)
                delta_phi_dft = scipy.fftpack.fft(np.angle(P_frames),axis=-1) * np.broadcast_to(F,P_frames.shape)
                delta_phi = np.real(scipy.fftpack.ifft(delta_phi_dft,axis=-1))[:,:,:T]
                P_frames = P_frames[:,:,:T]
                P_frames *= np.exp(alpha*np.complex(0,1)*delta_phi)
                Ps[d][n][k] = np.moveaxis(P_frames,-1,0)
                if verbose: pbar.update(1)
    if verbose: pbar.close()

    ## Inverse transform the edited frames
    print("Converting modified frames from pyramid")
    mframes = np.real(pyr2im(Ps,Rhs,Rls,verbose=verbose))

    return mframes

def get_temporal_filter(fs,fh,fl,length):
    if fh >= fs/2.:
        F = scipy.fftpack.fft(scipy.fftpack.ifftshift(scipy.signal.firwin(length,fl,fs=fs,pass_zero=False)))
    else:
        F = scipy.fftpack.fft(scipy.fftpack.ifftshift(scipy.signal.firwin(length,[fl,fh],fs=fs,pass_zero=False)))
    return F
