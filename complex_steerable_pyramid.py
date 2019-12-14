import numpy as np

def lowpass_filter(r,th):
    if np.pi/4 < r < np.pi/2
        return np.cos(np.pi/2*np.log2(4*r/np.pi))
    elif r <= np.pi/4:
        return 1
    else:
        return 0

def highpass_filter(r,th):
    if np.pi/4 < r < np.pi/2
        return np.cos(np.pi/2*np.log2(2*r/np.pi))
    elif r <= np.pi/4:
        return 0
    else:
        return 1

def angular_filter(r,th,k,K):
    if np.abs(th - np.pi*k/K) < np.pi/2:
        return np.factorial(K-1)/np.sqrt(K*np.factorial(2*(K-1))) * np.power(2*np.cos(th-np.pi*k/K),K-1)
    else:
        return 0

def bandpass_filter(r,th,n,N):
    return highpass(r/np.power(2,(N-n)/N),th) * lowpass(r/np.power(2,(N-n+1)/N),th)

def pyramid_filter(r,th,n,N,k,K):
    return bandpass_filter(r,th,n,N) * bandpass_filter(r,th,n,N)

def apply_filter(I,F):
    width = np.max(I.shape)

    w_y = scipy.fftpack.fftfreq(I.shape[0],d=2*np.pi*I.shape[0]/width)
    w_x = scipy.fftpack.fftfreq(I.shape[1],d=2*np.pi*I.shape[1]/width)
    W = np.stack((np.repeat(w_y.reshape(-1,1), I.shape[1], axis=1),np.repeat(w_x.reshape(1,-1), I.shape[0], axis=0)))
    
    R = np.norm(W,axis=2)
    Th = np.angle(W,axis=2)

    return I * np.vectorize(F)(R,Th)

def downsample2(im):
    pass

def upsample2(im,shape=None):
    pass


def im2pyr(im,D,N,K):
    I = scipy.fftpack.fft2(im)
    R_h = scipy.fftpack.ifft2(apply_filter(I,lambda r, th: highpass_filter(r/2,th)))
    P = []
    for d in range(D):
        P.append([ [ scipy.fftpack.ifft2(apply_filter(I,lambda r, th: pyramid_filter(r,th,n,N,k,K))) for k in range(K) ] for n in range(N) ])
        I = scipy.fftpack.fft2(downsample2(scipy.fftpack.ifft2(I)))
    R_l = scipy.fftpack.ifft2(I)
    return P, R_h, R_l

def pyr2im(P,R_h,R_l):
    pass