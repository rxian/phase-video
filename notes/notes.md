# Implementation Details

**Algorithm Overview** ([Wadhwa et al. 2013][1], Fig 2)
- Compute the complex steerable pyramid for each frame of the video (processed on $(x,y)$-planes, where the video sequence lies on $(x,y,t)$-hyperplane; now the changes in the phase component over time corresponds to motion).  [[details](#complex-steerable-pyramid)]
- Perform band-pass temporal filtering on the phase component of each image in the pyramid to isolate motion at specific frequencies (processed on $t$-axis, and now the amplitude of the resulting "image" corresponds to the amount of motion).  [[details](#temporal-filtering)]
- Smooth the "images" (optional).
- Multiply the resulting "images" by $\alpha$, and add it back to the phase component of the respective frames in the pyramid (positive coefficient gives motion amplification, and negative gives attenuation).
- Reconstruct video by collapsing the pyramid.

> **Algorithm (Motion Modification).**
> 
> Todo.

## Complex Steerable Pyramid

### Filters 

Performing filtering $F$ (defined in terms of polar frequency coordinates) to a DFT image $X$ (origin-centered with width $W$) can be written as $F\circ X$ (element-wise product), and the result is (definition of [atan2][3])

$$ Y_{i,j} = X_{i,j} \cdot F\left(\frac{\sqrt{i^2+j^2}}{W} \pi, \text{atan2}(j,i) \right). $$

The following filters are used to construct and collapse the pyramid ([Portilla et al. 2003][2], App I).  $L,H$ are low and high-pass filters, and $G_k$ ($k\in\{1,\cdots,K\}$) is used to choose the orientation (out of $K$ directions).  

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

We also use two windowing functions (in order), the first derived from above, and the second as proposed in ([Wadhwa et al. 2013][1], App A) that gives better results when the number of filters $N$ per octave exceeds 2.  For $n \in \{1,\cdots,N\}$,

$$W_n(r) = H(r/2^{(N-n)/N})\cdot L(r/2^{(N-n+1)/N}),$$

or

$$W_n(r) = TODO.$$

Finally, $B_{n,k}(r,\theta) = W_n(r)G_k(\theta)$ defines the filters used in the pyramid.

### Analysis

Given an image $I$, compute its DFT $\tilde I$, and then the pyramid as obtained follows with $\tilde I$ as input.

> **Algorithm (Pyramid Analysis).**
> 
> Input:
> - $\tilde I$ is a complex-valued DFT image of width $W$ centered at the origin.
> - $D$ is the depth of the pyramid.
> - $K$ is the number of filter orientations.
> - $N$ is the number of filters per octave.
> 
> Initialize:
> - $P$ is a $D\times N\times K$ list representing the pyramid.
> ---
> Compute $R_H \gets \tilde I\circ H(\bullet/2)$, which is the high-pass residual.
> 
> Let $J_0 := \tilde I$.
> 
> For $d=1,\cdots,D$:
> - For $n=1,\cdots,N$ and $k=1,\cdots,K$, store $P[d,n,k] \gets J_{d-1} \circ B_{n,k}$.
> - Set $J_d \gets J_{d-1} \circ L$, and then downsample by 2.
> 
> Set $R_L:= J_D$, which is the low-pass residual.
> 
> Return $P, R_H, R_L$.

### Synthesis

Given the pyramid $P$ of an image, reconstruct its DFT image $\tilde I$ as follows.  Note that by definition of the filters the complex conjugates are identity $\bar B_{n,k}=B_{n,k}$, $\bar H = H$, and $\bar L = L$.

> **Algorithm (Pyramid Synthesis).**
> 
> Inputs:
> - $P$ is a $D\times N\times K$ list representing the pyramid.
> - $R_H$, $R_L$ are the high and low-pass residuals.
> ---
> Let $\tilde I := R_H \circ \bar H(\bullet/2)$.
> 
> For $d,n,k=1$ to $D,N,K$ respectively:
> - Set $J$ to be the result of upsampling $P[d,n,k]$ by 2 for $d-1$ times.
> - $J \gets J \circ \bar B_{n,k}$.
> - Set $\bar J$ to be the complex conjugate of the reflection of $J$ about the $x$ and $y$-axis.
> - $\tilde I\gets \tilde I + J + \bar J$.
> 
> Add to $\tilde I$ the result of upsampling $(R_L\circ \bar L)$ by 2 for $d$ times.
> 
> Return $\tilde I$.

## Temporal Filtering

Todo.

[1]: http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf
[2]: https://www.cns.nyu.edu/pub/eero/portilla03-preprint-corrected.pdf
[3]: https://en.wikipedia.org/wiki/Atan2#Definition_and_computation
