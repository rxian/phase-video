# Implementation Details

**Algorithm Overview** ([Wadhwa et al. 2013][1], Fig. 2)
- Compute the complex steerable pyramid for each frame of the video (processed on $(x,y)$-planes, where the video sequence lies on $(x,y,t)$-hyperplane; now the changes in the phase component over time corresponds to motion).  [[details](#complex-steerable-pyramid)]
- Perform band-pass temporal filtering on the phase component of each image in the pyramid to isolate motion at specific frequencies (processed on $t$-axis, and now the amplitude of the resulting "image" corresponds to the amount of motion).  [[details](#temporal-filtering)]
- Smooth the "images" (optional, not described here).
- Multiply the resulting "images" by $\alpha$, and add it back to the phase component of the respective frames in the pyramid (positive coefficient gives motion amplification, and negative gives attenuation).  [[details](#motion-modification)]
- Reconstruct video by collapsing the pyramid.  [[details](#synthesis)]

> **Algorithm (Motion Modification).**
>  
> Input:
> - $I_{1:T}$ is a real-valued video sequence.
> - $D,K,N$ represents the depth, number of orientations and number of filters per octave to be used to construct the complex steerable pyramid.
> - $f_s$ represents the sampling rate of the video sequence, and $f_l,f_h$ represents the frequency range of the motion to be modified.
> - $\alpha$ is the magnification factor.
> - $B,F$ represent the types of filters to be used to construct the pyramid and for temporal filtering, respectively.
>  
> Initialize:
> - $P_{1:T}$, ${R_H}_{1:T}$, ${R_L}_{1:T}$ represent the pyramid sequence.
> - $J_{1:T}$ is the output video sequence.
> - Filter $F$ with $f_s,f_h,f_l$ as described in [temporal filtering section](#temporal-filtering).
> ---
> Obtain the complex steerable pyramid representation of each frame, i.e. $(P_t,{R_H}_t,{R_L}_t) \gets \text{PyramidAnalysis}(I_t,D,K,N,B)$ for all $t=1,\cdots T$, as described in [analysis algorithm](#analysis).
>  
> For $d,n,k=1$ to $D,N,K$ respectively:
> - Set $X_{1:T} := (P_1[d,n,k],P_2[d,n,k],\cdots,P_T[d,n,k])$, the evolution of $I$ in pyramid representation at scale $(D,N)$ and direction $K$.
> - Set $\Phi_{1:T}$ to be the phase component of $X_{1:T}$ ([`numpy.angle`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.angle.html)).
> - Set $\Delta\Phi_{1:T} \gets \text{TemporalFiltering}(\Phi_{1:T},F)$, performing temporal filtering on the phase, as described in [temporal filtering algorithm](#temporal-filtering).
> - For each $t$, perform motion modification as described in [motion modification section](#motion-modification) with $(X_t,\Delta\Phi_t,\alpha)$, and update $X_t$ to be the result.
>  
> Obtain the motion magnified video sequence $J_{1:T}$, where $J_t\gets \text{PyramidSynthesis}(P_t,{R_H}_t,{R_L}_t,D,K,N,B)$ for all $t$, as described in [synthesis algorithm](#synthesis).
>  
> Return $J_{1:T}$.

## Complex Steerable Pyramid

### Filters 

The result $Y$ of performing filtering $F$ to a DFT image $X$ in the frequency domain is an entry-wise product, $Y = F\circ X$.  If $F$ is defined in terms of polar frequency coordinates (i.e. its Fourier coefficients are a function of $(r,\theta)$) and $X$ is origin-centered with width $2W$ (i.e. DC component is at $(0,0)$), then the result is (cf. [atan2][3] definition)

$$ Y_{i,j} = X_{i,j} \cdot F\left(\frac{\sqrt{i^2+j^2}}{W} \pi, \text{atan2}(j,i) \right). $$

The following filters (defined using polar frequency coordinates) are used to construct and collapse the pyramid ([Portilla et al. 2000][2], App. I).  $L,H$ are low and high-pass filters, and $G_k$ ($k\in\{1,\cdots,K\}$) is used to choose the orientation (out of $K$ directions).  

$$
L(r) = \begin{cases}
\cos\left(\frac\pi2\log_2\frac{4r}{\pi}\right) & \text{if}\ \frac\pi4<r<\frac\pi2 \\
1 & \text{if}\ r\leq\frac\pi4\\
0 & \text{if}\ r\geq\frac\pi2,
\end{cases}
$$

$$
H(r) = \begin{cases}
\cos\left(\frac\pi2\log_2\frac{2r}{\pi}\right) & \text{if}\ \frac\pi4<r<\frac\pi2 \\
0 & \text{if}\ r\leq\frac\pi4\\
1 & \text{if}\ r\geq\frac\pi2,
\end{cases}
$$

$$
G_k(\theta) = \begin{cases} \frac{(K-1)!}{\sqrt{K(2(K-1))!}}\left(2\cos\left(\theta-\frac{\pi k}K\right)\right)^{K-1} &  \text{if } \left|\theta-\frac{\pi k}K\right| < \frac\pi2\\
0 & \text{else.}
\end{cases}
$$

We also define two windowing functions (or filters): the first is derived from above, and the second is proposed in ([Wadhwa et al. 2013][1], App. A) that gives better results when the number of filters $N$ per octave exceeds 2.  For $n \in \{1,\cdots,N\}$, the first filter is

$$W_n(r) = H(r/2^{(N-n)/N})\cdot L(r/2^{(N-n+1)/N}),$$

and the second is

$$W_n(r) = TODO.$$

Finally, $B_{n,k}(r,\theta) = W_n(r)G_k(\theta)$ defines the filters used in the pyramid.

### Analysis

Given an image $I$, obtain the complex steerable pyramid as follows.

> **Algorithm (Pyramid Analysis).**
> 
> Input:
> - $I$ is a real-valued square image of width $\ell$ centered at the origin.
> - $D$ is the depth of the pyramid.
> - $K$ is the number of filter orientations.
> - $N$ is the number of filters per octave.
> - $B$ is the filter to use.
> 
> Initialize:
> - $P$ is a $D\times N\times K$ list representing the pyramid.
> ---
> Compute $\tilde I \gets \text{DFT}(I)$.
>
> Compute $R_H \gets \text{IDFT}(\tilde I\circ H(\bullet/2))$, which is the high-pass residual.
> 
> Let $\tilde J_0 := \tilde I$.
> 
> For $d=1,\cdots,D$:
> - For $n=1,\cdots,N$ and $k=1,\cdots,K$, store $P[d,n,k] \gets \text{IDFT}(\tilde J_{d-1} \circ B_{n,k}$).
> - Set $J_d \gets \text{IDFT}(\tilde J_{d-1} \circ L)$, and downsample by 2.
> - Set $\tilde J_d \gets J_{d}$.
> 
> Set $R_L:= \text{IDFT}(\tilde J_D)$, which is the low-pass residual.
> 
> Return $P, R_H, R_L$.

### Synthesis

Given the complex steerabl pyramid representation $P$ of an image, reconstruct the original image $I$ as follows.  Note that by definition of the filters we use, their complex conjugates are identity, i.e. $\bar B_{n,k}=B_{n,k}$, $\bar H = H$, and $\bar L = L$.

> **Algorithm (Pyramid Synthesis).**
> 
> Inputs:
> - $P$ is a $D\times N\times K$ list representing the pyramid.
> - $R_H$, $R_L$ are the high and low-pass residuals.
> - $D$ is the depth of the pyramid.
> - $K$ is the number of filter orientations.
> - $N$ is the number of filters per octave.
> - $B$ is the filter to use.
> ---
> Let $\tilde I := \text{DFT}(R_H) \circ \bar H(\bullet/2)$.
> 
> For $d,n,k=1$ to $D,N,K$ respectively:
> - Upsample $P[d,n,k]$ by 2 for $d-1$ times, and set $\tilde J\gets \text{DFT}(P[d,n,k])$.
> - $\tilde J \gets \tilde J \circ \bar B_{n,k}$.
> - $\tilde I\gets \tilde I + \tilde J$.
> - Set $\tilde J$ to its complex conjugate, and reflect it about the $x$- and $y$-axis.
> - $\tilde I\gets \tilde I + \tilde J$.
> 
> Upsample $R_L$ by 2 for $d$ times, and set $\tilde I\gets \tilde I + \text{DFT}(R_L)$.
> 
> Return $\text{IDFT}(\tilde I)$.

## Temporal Filtering

Let the temporal filter $F$ be one of:
- FIR Window ([`scipy.signal.firwin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html)),
- Butterworth ([`scipy.signal.butter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html?highlight=butterworth)).

If the video is sampled at $f_s$, i.e. $f_s$ frames per second, then we can only recover motions that occur at at most $\frac{f_s}2$ by the Nyquist criterion.  Then to instantiate a (FIR window) band-pass filter that keeps signals between frequencies $[f_l,f_h]$ where $f_h\leq \frac{f_s}2$, we can use the following code snippet (width is the length of the filter; longer length results in less artifacts):

    F = scipy.fftpack.fft(scipy.fftpack.ifftshift(scipy.signal.firwin(width,[f_l,f_h],fs=f_s,pass_zero=False)))

where since time-domain impulse response of the filter is returned, DFT is taken to get the frequency response.

Let $I_{1:T}$ be a sequence of real-valued images, then temporal filtering is performed as below.

> **Algorithm (Temporal Filtering).**
>  
> Inputs:
> - $I_{1:T}$ a sequence of real-valued images.
> - $F$ is the frequency response of the temporal filter (1D) to use.
>  
> Initialize:
> - $J_{1:T}$ is the filtered image sequence of the same dimensions as $I_{1:T}$.
> ---
> For all pixel locations $i$:
> - Set $x:= (I_{1,i},I_{2,i},\cdots,I_{T,i})$, the evolution of the pixel at $i$.
> - Compute $\tilde x\gets \text{DFT}(x)$.
> - Compute $\tilde y\gets \tilde x \circ F$.
> - Set $J_{1:T,i} = \text{IDFT}(\tilde y)$.

## Motion Modification

Given a filtered frame $I$ in the pyramid (obtained with the [analysis algorithm](#analysis)), and an phase image $\Delta\Phi$ representing the motion present in this frame, we magnify the motion in the filtered frame by computing

$$
J = I \circ \exp(i(\alpha-1)\Delta\Phi),
$$

where $\exp$ is applied entry-wise, and $\alpha$ is the magnification factor: $\alpha>1$ is motion magnification, and $\alpha<1$ is motion attenuation.

[1]: http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf
[2]: https://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf
[3]: https://en.wikipedia.org/wiki/Atan2#Definition_and_computation