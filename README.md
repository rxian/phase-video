# Phase-Based Video Motion Processing

**See results, descriptions and implementation details [here](https://rxian.github.io/phase-video/).**

This is an incomplete Python re-implementation of the 2013 SIGGRAPH [paper][1] by Wadhwa, Rubinstein, Durand, and Freeman.  Code provided by original authors can be found [here](http://people.csail.mit.edu/nwadhwa/phase-video/).

Omitted features: (1) radial filter for quarter-octave pyramid, (2) smoothing of the temporally filtered phases, (3) attenuating motion in temporal stop-band, (4) handling videos with large motion.

## References 

Neal Wadhwa, Michael Rubinstein, Frédo Durand, and William T. Freeman. [Phase-based video motion processing][1]. _ACM Transactions on Graphics_, 2013.

Javier Portilla, and Eero P. Simoncelli. [A parametric texture model based on joint statistics of complex wavelet coefficients][2]. _International Journal of Computer Vision_, 2000.

[1]: http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf
[2]: https://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf
