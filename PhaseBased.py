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
    print("Converting to pyramid")
    if verbose: pbar = tqdm(total=T)
    for t in range(T):
        P, Rh, Rl = im2pyr(frames[t],D,N,K)
        Ps.append(P)
        Rhs.append(Rh)
        Rls.append(Rl)
        if verbose: pbar.update(1)
    if verbose: pbar.close()

    ## Motion editting
    print("Modifying motion")
    if verbose: pbar = tqdm(total=D*N*K)
    for d in range(D):
        for n in range(N):
            for k in range(K):
                P_frames = np.pad(np.array([x[d][n][k] for x in Ps],dtype=np.complex),((0,pad),(0,0),(0,0)),mode='edge')
                P_frames = np.moveaxis(P_frames,0,-1)
                delta_phi_dft = scipy.fftpack.fft(np.angle(P_frames),axis=-1) * np.broadcast_to(F,P_frames.shape)
                delta_phi = np.real(scipy.fftpack.ifft(delta_phi_dft,axis=-1))[:,:,:T]
                P_frames = P_frames[:,:,:T]
                P_frames *= np.exp(alpha*np.complex(0,1)*delta_phi)
                for t in range(T):
                    Ps[t][d][n][k] = P_frames[:,:,t]
                if verbose: pbar.update(1)
    if verbose: pbar.close()

    ## Inverse transform the edited frames
    print("Converting modified frames from pyramid")
    mframes = np.empty(frames.shape,dtype=np.float32)
    if verbose: pbar = tqdm(total=T)
    for t in range(T):
        mframes[t] = np.real(pyr2im(Ps[t],Rhs[t],Rls[t]))
        if verbose: pbar.update(1)
    if verbose: pbar.close()

    return mframes

def get_temporal_filter(fs,fh,fl,length):
    if fh == fs/2.:
        F = scipy.fftpack.fft(scipy.fftpack.ifftshift(scipy.signal.firwin(length,fl,fs=fs,pass_zero=False)))
    else:
        F = scipy.fftpack.fft(scipy.fftpack.ifftshift(scipy.signal.firwin(length,[fl,fh],fs=fs,pass_zero=False)))
    return F

def modify_motion_multiprocs(frames,alpha,D,N,K,F,num_workers=32,verbose=True):
    '''
    Perform phased-based motion processing.  Pyramid conversions are 
    parallelized.

    TODO: To be removed once im2pyr and pyr2im are optimized.

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
    @type  num_workers: integer
    @param num_workers: Number of parallel processes.
    @type  verbose: boolean
    @param verbose: Print progress bar.
    @rtype:   numpy.ndarray of shape (T,H,W)
    @return:  Edited frames.
    '''
    from multiprocessing import Queue, Process
    
    def partition(length,batch_size):
        ## Helper function to partition `list(range(length))`` into chunks that are at most `batch_size` long 
        dividers = list(range(0,length,batch_size)) + [length]
        return [range(dividers[i],dividers[i+1]) for i in range(len(dividers)-1)]
    
    T = len(frames)
    pad = len(F) - T
    Ps, Rhs, Rls = [None for t in range(T)], [None for t in range(T)], [None for t in range(T)]
 
    ## Transform frames to pyramid representation
    print("Converting to pyramid")
    if verbose: pbar = tqdm(total=T)
    for p in partition(T,num_workers):
        q = Queue()
        procs = [Process(target=im2pyr_multiprocs_wrapper, args=(np.array(frames[t]),t,D,N,K,q)) for t in p]
        for proc in procs: proc.start()
        for i in p:
            t, ret = q.get()
            P, Rh, Rl = ret
            Ps[t] = P
            Rhs[t] = Rh
            Rls[t] = Rl
            if verbose: pbar.update(1)
        for proc in procs: proc.join()
        q.close()
    if verbose: pbar.close()

    ## Motion editting (There are pickling issues with multiprocessing)
    print("Modifying motion")
    if verbose: pbar = tqdm(total=D*N*K)
    # for p in partition(D*N*K,num_workers):
    #     q = Queue()
    #     procs = []
    #     for t in p:
    #         d,n,k = np.unravel_index(t,(D,N,K))
    #         procs.append(Process(target=editing_multiprocs_wrapper, args=(np.array([x[d][n][k] for x in Ps],dtype=np.complex),alpha,F,d,n,k,q)))
    #     for proc in procs: proc.start()
    #     for i in p:
    #         d, n, k, P_frames = q.get()
    #         for t in range(T):
    #             Ps[t][d][n][k] = P_frames[t]
    #         if verbose: pbar.update(1)
    #     for proc in procs: proc.join()
    #     q.close()
    for d in range(D):
        for n in range(N):
            for k in range(K):
                P_frames = np.pad(np.array([x[d][n][k] for x in Ps],dtype=np.complex),((0,pad),(0,0),(0,0)),mode='edge')
                P_frames = np.moveaxis(P_frames,0,-1)
                delta_phi_dft = scipy.fftpack.fft(np.angle(P_frames),axis=-1) * np.broadcast_to(F,P_frames.shape)
                delta_phi = np.real(scipy.fftpack.ifft(delta_phi_dft,axis=-1))[:,:,:T]
                P_frames = P_frames[:,:,:T]
                P_frames *= np.exp(alpha*np.complex(0,1)*delta_phi)
                for t in range(T):
                    Ps[t][d][n][k] = P_frames[:,:,t]
                if verbose: pbar.update(1)
    if verbose: pbar.close()

    ## Inverse transform the edited frames
    mframes = np.empty(frames.shape,dtype=np.float32)
    print("Converting modified frames from pyramid")
    if verbose: pbar = tqdm(total=T)
    for p in partition(T,num_workers):
        q = Queue()
        procs = [Process(target=pyr2im_multiprocs_wrapper, args=(Ps[t],Rhs[t],Rls[t],t,q)) for t in p]
        for proc in procs: proc.start()
        for i in p:
            t, frame = q.get()
            mframes[t] = np.real(frame)
            if verbose: pbar.update(1)
        for proc in procs: proc.join()
        q.close()
    if verbose: pbar.close()

    return mframes

def im2pyr_multiprocs_wrapper(im,t,D,N,K,q):
    q.put((t,im2pyr(im,D,N,K)))

def pyr2im_multiprocs_wrapper(P,Rh,Rl,t,q):
    q.put((t,pyr2im(P,Rh,Rl)))

def editing_multiprocs_wrapper(P_frames,alpha,F,d,n,k,q):
    T = len(P_frames)
    pad = len(F) - T
    P_frames = np.pad(P_frames,((0,pad),(0,0),(0,0)),mode='edge')
    P_frames = np.moveaxis(P_frames,0,-1)
    delta_phi_dft = scipy.fftpack.fft(np.angle(P_frames),axis=-1) * np.broadcast_to(F,P_frames.shape)
    delta_phi = np.real(scipy.fftpack.ifft(delta_phi_dft,axis=-1))[:,:,:T]
    P_frames = P_frames[:,:,:T]
    P_frames *= np.exp(alpha*np.complex(0,1)*delta_phi)
    P_frames = np.moveaxis(P_frames,-1,0)
    q.put((d,n,k,P_frames))
