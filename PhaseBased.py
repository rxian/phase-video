import numpy as np
import scipy.fftpack, scipy.signal
from ComplexSteerablePyramid import pyr2im, im2pyr

def modify_motion(frames,alpha,D,N,K,fs,fl,fh):
    T = len(frames)
    pad = 0 if T%2 == 1 else 1
    Ps = []
    Rhs, Rls = [], []

    if fh == fs/2.:
        temporal_filter = scipy.fftpack.fft(scipy.fftpack.ifftshift(scipy.signal.firwin(T+pad,fl,fs=fs,pass_zero=False)))
    else:
        temporal_filter = scipy.fftpack.fft(scipy.fftpack.ifftshift(scipy.signal.firwin(T+pad,[fl,fh],fs=fs,pass_zero=False)))

    for t in range(T):
        P, Rh, Rl = im2pyr(frames[t],D,N,K)
        Ps.append(P)
        Rhs.append(Rh)
        Rls.append(Rl)

    for d in range(D):
        for n in range(N):
            for k in range(K):

                # P_frames will have shape (H, W, T+pad), as opposed to (T+pad, H, W)
                P_frames = np.pad(np.array([x[d][n][k] for x in Ps],dtype=np.complex),((0,pad),(0,0),(0,0)),mode='edge')
                P_frames = np.moveaxis(P_frames,0,-1)

                delta_phi_dft = scipy.fftpack.fft(np.angle(P_frames),axis=-1) * np.broadcast_to(temporal_filter,P_frames.shape)
                delta_phi = np.real(scipy.fftpack.ifft(delta_phi_dft,axis=-1))

                P_frames *= np.exp((alpha-1)*np.complex(0,1)*delta_phi)

                for t in range(T):
                    Ps[t][d][n][k] = P_frames[:,:,t]

    ret = []
    for t in range(T):
        ret.append(np.real( pyr2im(Ps[t],Rhs[t],Rls[t]) ))
    
    return np.array(ret)


## Debug code below
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    frame1 = np.zeros((50,50),dtype=np.float32)
    frame1[25:28,25:28] = 255

    frames = np.empty((30,50,50))
    for i in range(len(frames)):
        if i % 2:
            frames[i] = frame1
        else:
            frames[i] = np.roll(frame1,1,axis=0)

    plt.figure(); plt.imshow(frames[0])
    plt.figure(); plt.imshow(frames[1]-frames[0])

    mframes = modify_motion(frames, 2, 2, 1, 4, 1, 0.2, 0.5)

    plt.figure(); plt.imshow(mframes[0].clip(0,255));plt.colorbar()
    plt.figure(); plt.imshow(mframes[1].clip(0,255))
    # plt.figure(); plt.imshow(frames[1]-frames[0])
    plt.figure(); plt.imshow(mframes[1]-frames[0]);plt.colorbar()

