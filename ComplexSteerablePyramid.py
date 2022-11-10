import numpy as np
import scipy.fftpack
from tqdm import tqdm

def im2pyr(im,D,N,K,verbose=False):
    '''
    Transform an image to complex steerable pyramid representation.

    @type  im: real-valued numpy.ndarray of shape (B,H,W)
    @param im: Batched images to be transformed.
    @type  D: integer
    @param D: Depth of pyramid (number of octaves).
    @type  N: integer
    @param N: Number of suboctaves per octave.
    @type  K: integer
    @param K: Number of pyramid orientations.
    @rtype:   (P, Rh, Rl) 3-tuple; P is a nested list of shape 
              (D,N,K), Rh and Rl are numpy.ndarrays of 2D
    @return:  P stores the images in the pyramid; Rh and Rl are
              highpass and lowpass residuals.
    '''
    dft = scipy.fftpack.fft2
    idft = scipy.fftpack.ifft2

    if verbose: pbar = tqdm(total=D*N*K)

    I = dft(im)
    Rh = idft(apply_filter(I,lambda r, th: highpass_filter(r/2.,th)))
    P = []
    for d in range(D):
        this_D = []
        for n in range(N):
            this_n = []
            for k in range(K):
                this_n.append(idft(apply_filter(I,lambda r, th: pyramid_filter(r,th,n,N,k,K))))
                if verbose: pbar.update(1)
            this_D.append(this_n)
        P.append(this_D)
        I = downsample2(apply_filter(I,lowpass_filter))
    Rl = idft(I)

    if verbose: pbar.close()
    return P, Rh, Rl

def pyr2im(P,Rh,Rl,verbose=False):
    '''
    Transform an image from complex steerable pyramid representation.

    @type  P: nested list
    @param P: Images in the pyramid
    @type  Rh: numpy.ndarray
    @param Rh: highpass residual
    @type  Rl: numpy.ndarray
    @param Rl: lowpass residual
    @rtype:   numpy.ndarray
    @return:  Reconstructed image.
    '''
    dft = scipy.fftpack.fft2
    idft = scipy.fftpack.ifft2

    D = len(P)
    N = len(P[0])
    K = len(P[0][0])

    if verbose: pbar = tqdm(total=D*N*K)

    I = dft(Rl)
    for d in range(D-1,-1,-1):
        I = apply_filter(upsample2(I,shape=P[d][0][0].shape[-2:]),lowpass_filter)
        for n in range(N):
            for k in range(K):
                J = apply_filter(dft(P[d][n][k]),lambda r, th: pyramid_filter(r,th,n,N,k,K))
                I += J
                J_c = np.flip(scipy.fftpack.fftshift(np.conjugate(J),(-2,-1)),(-2,-1))
                if J_c.shape[-2] % 2 == 0:
                    J_c = np.roll(J_c,1,axis=-2)
                    J_c[:,0,:] = 0.
                if J_c.shape[-1] % 2 == 0:
                    J_c = np.roll(J_c,1,axis=-1)
                    J_c[:,:,0] = 0.
                J_c = scipy.fftpack.ifftshift(J_c,(-2,-1))
                I += J_c
                if verbose: pbar.update(1)
    I += apply_filter(dft(Rh),lambda r, th: highpass_filter(r/2.,th))

    if verbose: pbar.close()
    return idft(I)

def lowpass_filter(r,th):
    '''
    Returns the Fourier coefficient of a lowpass filter.
    '''
    if np.pi/4. < r < np.pi/2.:
        return np.cos(np.pi/2.*np.log2(4.*r/np.pi))
    elif r <= np.pi/4.:
        return 1.
    else:
        return 0.

def highpass_filter(r,th):
    '''
    Returns the Fourier coefficient of a highpass filter.
    '''
    if np.pi/4. < r < np.pi/2.:
        return np.cos(np.pi/2.*np.log2(2.*r/np.pi))
    elif r <= np.pi/4.:
        return 0.
    else:
        return 1.

def angular_filter(r,th,k,K):
    '''
    Returns the Fourier coefficient of an angular filter.
    '''
    c = np.math.factorial(K-1)/np.sqrt(K*np.math.factorial(2*(K-1)))
    angle = np.min((np.abs(th-np.pi*k/K),2.*np.pi-np.abs(th-np.pi*k/K)))
    if angle < np.pi/2.:
        return c*np.power(2*np.cos(angle),K-1)
    else:
        return 0.

def bandpass_filter(r,th,n,N):
    '''
    Returns the Fourier coefficient of a bandpass filter.
    '''
    return highpass_filter(r/np.power(2.,(N-n-1)/N),th) * lowpass_filter(r/np.power(2.,(N-n)/N),th)

def pyramid_filter(r,th,n,N,k,K):
    '''
    Returns the Fourier coefficient of a pyramid filter.
    '''
    return bandpass_filter(r,th,n,N) * angular_filter(r,th,k,K)

def get_polar_coors(h,w,stretch=False):
    '''
    Returns two matrices representing (radius, angle) pairs.
    '''
    length = max(h,w)
    d_y = 1/(2.*np.pi) if stretch else 1/(2.*np.pi*h/length)
    d_x = 1/(2.*np.pi) if stretch else 1/(2.*np.pi*w/length)
    w_y = scipy.fftpack.fftfreq(h,d=d_y)
    w_x = scipy.fftpack.fftfreq(w,d=d_x)
    W = np.stack((np.repeat(w_y.reshape(-1,1),w,axis=1),np.repeat(w_x.reshape(1,-1),h,axis=0)))
    R = np.linalg.norm(W,axis=0) 
    Th = np.arctan2(W[0],W[1])
    return R, Th

def get_filter_coeffs(h,w,F,stretch=False):
    '''
    Returns filter coefficients for an image of shape (h,w).
    '''
    R, Th = get_polar_coors(h,w,stretch)
    return np.vectorize(F)(R,Th)

def apply_filter(I,filt,stretch=False):
    '''
    Apply filter F to I in frequency domain.
    '''
    F = get_filter_coeffs(I.shape[-2],I.shape[-1],filt,stretch)
    F = np.broadcast_to(F,I.shape)
    return I * F

def downsample2(I):
    '''
    Downsample a lowpassed image by 2 in frequency domain.
    '''
    H = I.shape[-2]
    W = I.shape[-1]
    window_left = lambda width_big, width_small: np.where(scipy.fftpack.fftshift(scipy.fftpack.fftfreq(width_big)) == 0)[0].flatten()[0] - np.where(scipy.fftpack.fftshift(scipy.fftpack.fftfreq(width_small)) == 0)[0].flatten()[0]
    new_h = int(np.ceil(H/2.))
    new_w = int(np.ceil(W/2.))
    offset_y = window_left(H,new_h)
    offset_x = window_left(W,new_w)
    return scipy.fftpack.ifftshift(scipy.fftpack.fftshift(I,(-2,-1))[:,offset_y:offset_y+new_h,offset_x:offset_x+new_w],(-2,-1))
 
def upsample2(I,shape=None):
    '''
    Upsample an image by 2 in frequency domain.
    '''
    H = I.shape[-2]
    W = I.shape[-1]
    window_left = lambda width_big, width_small: np.where(scipy.fftpack.fftshift(scipy.fftpack.fftfreq(width_big)) == 0)[0].flatten()[0] - np.where(scipy.fftpack.fftshift(scipy.fftpack.fftfreq(width_small)) == 0)[0].flatten()[0]
    new_h = H * 2 if shape is None else shape[0]
    new_w = W * 2 if shape is None else shape[1]
    offset_y = window_left(new_h,H)
    offset_x = window_left(new_w,W)
    ret = np.zeros((I.shape[0],new_h,new_w),dtype=np.complex)
    ret[:,offset_y:offset_y+H,offset_x:offset_x+W] = scipy.fftpack.fftshift(I,(-2,-1))
    return scipy.fftpack.ifftshift(ret,(-2,-1))
