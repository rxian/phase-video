#%%
import numpy as np 
import scipy
import scipy.fftpack
import matplotlib.pyplot as plt

#%%

def osc_rects(t,sep=3,disp=2,width=3,length=23):
    offset = lambda t: t%disp if (t%(disp*2))/disp < 1 else disp - t%disp
    x = np.zeros((t,length),dtype=np.float32)
    for i in range(t):
        r1 = length//2 - sep - offset(i); l1 = r1 - width + 1
        x[i,l1:r1+1] = x[i,length-r1-1:length-l1] = 1
    return x

# def trans_sin(t,length=23):


duration = 2
length = 40

x_orig = osc_rects(duration,sep=7,width=4,length=length)

# x_orig = np.empty((duration,length),dtype=np.float32)
# x_orig[0] = np.sin(np.arange(23)*2*np.pi/12)
# x_orig[1] = np.roll(x_orig[0],1)

# x_orig = np.zeros((duration,length),dtype=np.float32)
# x_orig[0,10:15] = 1
# x_orig[1] = np.roll(x_orig[0],1)
# x_orig[:,2:5] = 1


plt.figure()
for x in x_orig:
    plt.plot(x)

#%%

f_orig = scipy.fftpack.fft(x_orig,axis=1)


x_mag = np.zeros((duration,length),dtype=np.float32)
x_mag[0] = x_orig[0]

num_partitions = 2

reflect_w = np.vectorize(lambda i : -i % length if ( length%2==1 or (length%2==0 and i%length!=length//2) ) else i % length)

# Filter using the smallest window (1 pixel)
for t in range(1,duration):
    this_f_mag = np.array(f_orig[t],copy=True)

    partitions = np.array_split(np.argwhere(scipy.fftpack.fftfreq(length)<0).flatten(),num_partitions)

    for p in partitions:
        temp = np.zeros(length,dtype=np.complex)

        temp[p] = f_orig[t-1,p]
        last_f_filt = np.array(temp,copy=True)
        last_x_filt = scipy.fftpack.ifft(last_f_filt)
        last_phase = np.angle(last_x_filt)

        temp[p] = f_orig[t,p]
        this_f_filt = np.array(temp,copy=True)
        this_x_filt = scipy.fftpack.ifft(this_f_filt)
        this_phase = np.angle(this_x_filt)

        # print(last_f_filt,this_f_filt)

        phase_diff = this_phase - last_phase

        mag_const = 1
        x_mag_filt = this_x_filt * np.exp(mag_const * np.complex(0,1) * phase_diff)
        f_mag_filt = scipy.fftpack.fft(x_mag_filt)

        this_f_mag[reflect_w(p)] = np.conjugate(f_mag_filt[p])
        this_f_mag[p] = f_mag_filt[p]

        # reflect_p = 
        # if reflect_w(i,length) != i:
        #     this_f_mag[reflect_w(i,length)] = np.conjugate(f_mag_filt[i])

    # print(f_orig[t])
    # print(this_f_mag)
    # print(f_orig[t]-this_f_mag)

    x_mag[t] = np.real(scipy.fftpack.ifft(this_f_mag))

plt.figure()
for x in x_mag:
    plt.plot(x)
plt.plot(x_orig[-1])

# plt.figure(); plt.plot(f1); plt.plot(f2); plt.plot(f3)






#%%

i = np.argwhere(scipy.fftpack.fftfreq(g.shape[1]) < 0).flatten()
g[:,i] = 0

#%%

ff = scipy.fftpack.ifft(g,axis=1)

#%%

for i in range(ff.shape[0]):
    fig, ax = plt.subplots(1,2)
    ax[0].plot(np.abs(ff[i]))
    ax[1].plot(np.angle(ff[i]))

#%%

test = ff[1] * np.exp(1*2*np.pi*scipy.fftpack.fftfreq(g.shape[1])*np.complex(0,1)*(np.angle(ff[1])-np.angle(ff[0])))

ttest = scipy.fftpack.fft(test)
ttest[np.argwhere(scipy.fftpack.fftfreq(g.shape[1]) < 0).flatten()] = np.conjugate(np.flip(ttest[np.argwhere(scipy.fftpack.fftfreq(g.shape[1]) > 0).flatten()]))
tttest = scipy.fftpack.ifft(ttest)
plt.figure(); plt.plot(tttest); plt.plot(f[1])

# %%

g1 = scipy.fft(f1)
g2 = scipy.fft(f2)
g3 = scipy.fft(f3)

d1 = np.angle(g1)-np.angle(g2)
d2 = np.angle(g2)-np.angle(g3)

plt.figure(); plt.plot(d1)
plt.figure(); plt.plot(d2)
# plt.figure(); plt.plot(np.angle(g3))


# %%

plt.figure(); plt.plot(np.abs(g1)); plt.plot(np.abs(g2)); plt.plot(np.abs(g3))

# %%

plt.figure(); plt.plot(f2); plt.plot( f2 * np.exp(np.complex(0,1)*np.arange(21)*d1*3) )


# %%
plt.plot(scipy.fftpack.fftshift(np.abs(g1)))
plt.plot(np.abs(g1))
scipy.fftpack.fftshift(np.abs(g1))[0]

# %%
# %%
x = np.arange(24)
f1 = np.sin(x*2*np.pi/12)
f2 = np.roll(f1,1)


g1=scipy.fft(f1)
gg1 = np.array(g1,copy=True)
gg1[1:12] = 0
ff1 = scipy.ifft(gg1)

g2=scipy.fft(f2)
gg2 = np.array(g2,copy=True)
gg2[1:12] = 0
ff2 = scipy.ifft(gg2)

fig, ax = plt.subplots(1,2); ax[0].plot(np.abs(ff1)); ax[1].plot(np.angle(ff1))
fig, ax = plt.subplots(1,2); ax[0].plot(np.abs(ff2)); ax[1].plot(np.angle(ff2))

#%%

test = ff2 * np.exp(4*np.complex(0,1)*(np.angle(ff2)-np.angle(ff1)))

ttest = scipy.fftpack.fft(test)
ttest[1:12] = np.conjugate(np.flip(ttest[13:]))
tttest = scipy.fftpack.ifft(ttest)
plt.figure(); plt.plot(tttest); plt.plot(f2); plt.plot(f1)

# %%
