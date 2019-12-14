#%%
import numpy as np
import scipy.fftpack

def lowpass_filter(r,th):
    if np.pi/4. < r < np.pi/2.:
        return np.cos(np.pi/2.*np.log2(4.*r/np.pi))
    elif r <= np.pi/4.:
        return 1.
    else:
        return 0.

def highpass_filter(r,th):
    # print(r,th)
    if np.pi/4. < r < np.pi/2.:
        return np.cos(np.pi/2.*np.log2(2.*r/np.pi))
    elif r <= np.pi/4.:
        return 0.
    else:
        return 1.

def angular_filter(r,th,k,K):
    c = np.math.factorial(K-1)/np.sqrt(K*np.math.factorial(2.*(K-1)))
    angle = np.min((np.abs(th-np.pi*k/K),2.*np.pi-np.abs(th-np.pi*k/K)))
    if angle < np.pi/2.:
        return c*np.power(2*np.cos(angle),K-1)
    else:
        return 0.

def bandpass_filter(r,th,n,N):
    return highpass_filter(r/np.power(2.,(N-n-1)/N),th) * lowpass_filter(r/np.power(2.,(N-n)/N),th)

def pyramid_filter(r,th,n,N,k,K):
    return bandpass_filter(r,th,n,N) * angular_filter(r,th,k,K)

def apply_filter(I,F):
    width = np.max(I.shape)

    w_y = scipy.fftpack.fftfreq(I.shape[0],d=1/(2.*np.pi*I.shape[0]/width))
    w_x = scipy.fftpack.fftfreq(I.shape[1],d=1/(2.*np.pi*I.shape[1]/width))
    W = np.stack((np.repeat(w_y.reshape(-1,1), I.shape[1], axis=1),np.repeat(w_x.reshape(1,-1), I.shape[0], axis=0)))
    
    R = np.linalg.norm(W,axis=0) 
    Th = np.arctan2(W[0],W[1])

    return I * np.vectorize(F)(R,Th)

def downsample2(im):
    return np.array(im[::2,::2],copy=True)

def upsample2(im,shape=None):
    ret = np.array(im,copy=True)
    ret = np.insert(ret,range(1,im.shape[0]),(ret[:-1]+ret[1:])/2.,axis=0)
    ret = np.insert(ret,range(1,im.shape[1]),(ret[:,:-1]+ret[:,1:])/2.,axis=1)
    ret = np.pad(ret,((0,1),(0,1)), mode='edge')
    if shape is not None:
        return ret[:shape[0],:shape[1]]
    else:
        return ret

def im2pyr(im,D,N,K):
    dft = scipy.fftpack.fft2
    idft = scipy.fftpack.ifft2

    I = dft(im)
    R_h = idft(apply_filter(I,lambda r, th: highpass_filter(r/2.,th)))
    P = []
    for d in range(D):
        P.append([ [ idft(apply_filter(I,lambda r, th: pyramid_filter(r,th,n,N,k,K))) for k in range(K) ] for n in range(N) ])
        I = dft(downsample2(idft(apply_filter(I,lowpass_filter))))
    R_l = idft(I)
    return P, R_h, R_l

def pyr2im(P,R_h,R_l):
    dft = scipy.fftpack.fft2
    idft = scipy.fftpack.ifft2

    D = len(P)
    N = len(P[0])
    K = len(P[0][0])

    I = dft(R_l)
    for d in range(D-1,-1,-1):
        I = apply_filter(dft(upsample2(idft(I)),shape=P[d][0][0].shape),lowpass_filter)
        for n in range(N):
            for k in range(K):
                J = apply_filter(dft(P[d][n][k]),lambda r, th: pyramid_filter(r,th,n,N,k,K))
                J_c = np.flip(scipy.fftpack.fftshift(np.array(np.conjugate(J),copy=True)),axis=(0,1))
                if J_c.shape[0] % 2 == 0:
                    J_c = np.roll(J_c,1,axis=0)
                    J_c[0,:] = 0.
                if J_c.shape[1] % 2 == 0:
                    J_c = np.roll(J_c,1,axis=1)
                    J_c[:,0] = 0.
                J_c = scipy.fftpack.ifftshift(J_c)
                I += J + J_c
    I += apply_filter(dft(R_h),lambda r, th: highpass_filter(r/2.,th))
    return idft(I)




#%%%%%%%%%%%%%%%%%%
# DEBUG CODE BELOW

import matplotlib.pyplot as plt

#%%
disc = np.zeros((100,100),dtype=np.float32)
for i in range(disc.shape[0]):
    for j in range(disc.shape[1]):
        if np.sqrt((i-disc.shape[0]//2)**2+(j-disc.shape[1]//2)**2) < 25:
            disc[i,j] = 1

im = disc
plt.imshow(disc)

#%%
from PIL import Image
import requests
from io import BytesIO

# io.BytesIO(urllib.urlopen("https://rxian2.web.illinois.edu/cs445/proj1/a_im_in_colored2.jpg").read())
im = Image.open(BytesIO(requests.get("https://rxian2.web.illinois.edu/cs445/proj1/a_im_in_colored2.jpg").content)).convert('LA')
im = np.array(im,dtype=np.float32)[:,:,0]
plt.imshow(im);plt.colorbar();

#%%
P, R_h, R_l = im2pyr(im,2,2,4)

#%%
re = np.real(pyr2im(P, R_h, R_l))
plt.imshow(np.real(re));plt.colorbar();

# %%
plt.imshow(np.real(re)-im);plt.colorbar();

# %%
