# Paper Summary

This is an incomplete summary of ([Wadhwa et al. 2013][1], Sec. 3), with added details for rationalizing the methodology.

The motivation for phase-based video motion processing comes from the Fourier shift theorem.  Therefore, understanding the motion processing method used in this paper requires knowledge of Fourier analysis.  

We first represent an image with the Fourier synthesis formula.  Let $f\in\mathbb{R}^{[0,1]}$ denote a 1D image of unit length, where its values can be interpreted as luminance, then it can be written as a trigonometric polynomial using the definition of Fourier series, i.e.

$$f(x) = \sum_{k=-\infty}^\infty c_k \exp\left(i2\pi kx\right), $$

where $c_k$ is some complex number.  

## Motion and Phase

The Fourier shift theorem states that 

$$f(x+\delta(t)) = \sum_{k=-\infty}^\infty c_k \exp\left(i2\pi k(x+\delta(t))\right),$$

meaning that if the image is translated by $\delta(t)$ over time, then it will appear as a phase shift on the trigonometric polynomial.  If we take the phase component of $f(x+\delta(t))$, and use a temporal filter with zero DC response to filter out the changing phase that corresponds to the translation, i.e. to extract

$$\Delta\phi(t)=2\pi k \delta(t),$$

then we can directly magnify or attenuate this translation by increasing or decreasing the change in phase via multiplication, i.e.

$$f'(x+\delta(t)) = \sum_{k=-\infty}^\infty c_k \exp\left(i2\pi k(x+\delta(t))\right)\cdot \exp\left(i\alpha\Delta\phi(t)\right),$$

with $\alpha > 1$ resulting in magnification and $\alpha < 1$ in attenuation.  Refer to [`docs/examples.ipynb`](./examples.ipynb) for toy examples of motion modification.

However, most interesting motions are not just translations.  Usually, motions are localized and we have to deal with $\delta(x,t)$.  Also, they would also require multiple frequency bands to represent, meaning that by indexing the motions by $j$, we need to treat the phases over frequency bands $k\in A_j$ as a whole.  Now, the motion is written as

$$f(x+\delta(x,t)) = \sum_{j} \left(\sum_{k\in A_j} c_k \exp\left(i2\pi kx\right) \right) \cdot \exp
\left(i2\pi \sum_{k\in A_j}\delta_j(x,t,k)\right),$$

and we process each summand as a whose with temporal filters of appropriate pass-bands to isolate the motion of interest.

## Complex Steerable Pyramid

Without further assumption, a simple method to partition the 2D frequency spectrum for motion extraction is the steerable pyramid.  Ideally, the pyramid divides the 2D frequency domain into concentric rings and sectors of equal angles (cf. [Wadhwa et al. 2013][1], Fig. 4).


Each partition captures objects of a certain size and orientation, hence motions of a certain scale and direction.  The corners not included in the circle is the high-pass residual, and the center part that the pyramid of finite depth left un-partitioned is the low-pass residual.  

Since for real-valued images, Fourier coefficients are conjugate symmetric, so we only need to keep half of the 2D frequency spectrum to perform processing on.  On the other hand, we have to throw out the other half anyways, because the sum of each partition and its conjugate reflection gives a real-valued time-domain signal without the zero phase.  This gives the "complex" nature of this pyramid.

Finally, note that as analyzed in ([Wadhwa et al. 2013][1], Sec. 3.2), there is a trade-off between the amount of motion magnification that can be achieved, which means using a pyramid with more levels and orientations, and the amount of distortion in the output, which is minimized when using a pyramid with fewer levels and orientations.

[1]: http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf
[2]: https://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf
