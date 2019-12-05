# Implementation Details

**Algorithm Overview** ([Wadhwa et al. 2013][1], Fig 2)
- Compute the complex steerable pyramid for each frame of the video (processed on $(x,y)$-planes, where the video sequence lies on $(x,y,t)$-hyperplane; now the changes in the phase component over time corresponds to motion).  [[details](#complex-steerable-pyramid)]
- Perform band-pass temporal filtering on the phase component of each image in the pyramid to isolate motion at specific frequencies (processed on $t$-axis, and now the amplitude of the resulting "image" corresponds to the amount of motion).  [[details](#temporal-filtering)]
- (Optional).  Smooth the "images".
- Multiply the resulting "images" by $\alpha$, and add it back to the phase component of the respective frames in the pyramid (positive coefficient gives motion amplification, and negative gives attenuation; some temporal interpolation could be done between time windows?).  [[details](#temporal-filtering)]
- Reconstruct video by collapsing the pyramid.  [[details](#synthesis)]

---
**Algorithm (Motion Modification).**

Todo.

---

## Complex Steerable Pyramid

### Filters 

Performing filtering $F$ (defined in terms of frequency polar coordinates) to a DFT image $X$ (origin-centered with width $W$) can be written as $F\circ X$ (element-wise product), and the result is (definition of [atan2][3])

$$ Y_{i,j} = X_{i,j} \cdot F\left(\frac{\sqrt{i^2+j^2}}{W} \pi, \text{atan2}(j,i) \right). $$

The following filters are used to construct and collapse the pyramid ([Portilla et al. 2000][2], Sec 2.1).  $L,H$ are low and high-pass filters, and $G_k$ ($k\in\{1,\cdots,K\}$) is used to choose the orientation (out of $K$ directions).  

$$
L(r) = \begin{cases}
2\cos\left(\frac\pi2\log_2\frac{4r}{\pi}\right) & \text{if}\ \frac\pi4<r<\frac\pi2 \\
2 & \text{if}\ r\leq\frac\pi4\\
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
G_k(\theta) = \begin{cases}
\frac{2^{K-1}(K-1)!}{\sqrt{K(2(K-1))!}}\cos\left(\theta-\frac{\pi k}K\right)^{K-1} & \text{if}\ \left|\theta-\frac{\pi k}K\right|<\frac\pi2 \\
0 & \text{else}.
\end{cases}
$$

We also use two windowing functions (in order), the first derived from above, and the second as proposed in ([Wadhwa et al. 2013][1], App A) that gives better results when the number of filters $N$ per octave exceeds 2.  For $n \in \{1,\cdots,N\}$,

$$W_n(r) = H(r/2^{(N-n)/N})\cdot \frac12L(r/2^{(N-n+1)/N}),$$

or

$$W_n(r) = TODO.$$

Finally, $B_{n,k}(r,\theta) = W_n(r)G_k(\theta)$ defines the filters used in the pyramid.


### Analysis

Given an real-valued image $I$, compute its DFT $\tilde I$, and then decompose it into amplitude $A$ and phase $\Phi$ components, namely $\tilde I=A\circ e^{i\Phi}$.  Next, construct the phase pyramid as follows with $\Phi$ as input.

---
**Algorithm (Phase Pyramid Analysis).**

Input:
- $\Phi$ is a real-valued image, representing the phase of the complex-valued DFT image of width $W$ centered at the origin.
- $D$ is the depth of the pyramid.
- $K$ is the number of filter orientations.
- $N$ is the number of filters per octave.

Initialize:
- $P$ is a $D\times N\times K$ list representing the phase pyramid.

Compute $R_H \gets \Phi\circ H(\cdot/2)$, which is the high-pass residual.

Set $J_0 := \Phi$.

For $d=1,\cdots,D$:
- For $n=1,\cdots,N$ and $k=1,\cdots,K$, store $P[d,n,k] \gets J_{d-1} \circ B_{n,k}$.
- Compute $J_d \gets J_{d-1} \circ L$, and then decimate it by 2.

Set $R_L:= J_D$, which is the low-pass residual.

Return $P, R_H, R_L$.

---

### Synthesis

Given the phase pyramid $P$ of a real-valued image, reconstruct the phase component $\Phi$ of its DFT with the following algorithm, and then synthesize the image with the DFT amplitude $A$ by applying the inverse DFT to $\tilde I=Ae^{i\Phi}$.

---
**Algorithm (Phase Pyramid Synthesis).**

Inputs:
- $P$ is a $D\times N\times K$ list representing the phase pyramid.
- $R_H$, $R_L$ are high and low-pass residuals.

Let $\Phi \gets R_H$.

For $d,n,k=1$ to $D,N,K$ respectively:
- Set $J$ to be the result of upsampling $P[d,n,k]$ by 2 for $d-1$ times (can use bilinear interpolation).
- Set $\bar J$ to be the result of flipping $J$ horizontally and vertically.
- $\Phi  \gets \Phi + J - \bar J$.

Add to $\Phi$ the result of upsampling $R_L$ by 2 for $d$ times.

Return $\Phi$.

---

## Temporal Filtering

Todo.

[1]: http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf
[2]: https://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf
[3]: https://en.wikipedia.org/wiki/Atan2#Definition_and_computation
