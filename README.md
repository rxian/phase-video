# Phase-Based Video Motion Processing

This is an (so far) incomplete (and un-optimized) Python re-implementation of the 2013 SIGGRAPH [paper][1] by Wadhwa, Rubinstein, Durand, and Freeman.  Code provided by original authors can be found [here](http://people.csail.mit.edu/nwadhwa/phase-video/).

A summary of the paper and details of implementation can be found under [`docs`](./docs), and results are posted [here](https://rxian2.web.illinois.edu/cs445/project/).

## Omitted Features
- Radial filter for quarter-octave pyramid.
- Smoothing of the temporally filtered phases.
- Attenuating motion in temporal stop-band.
- Handling videos with large motion.

## References 

Wadhwa, Neal, Michael Rubinstein, Frédo Durand, and William T. Freeman. ["Phase-based video motion processing."][1] _ACM Transactions on Graphics (TOG)_ 32, no. 4 (2013): 80.
Harvard	

Portilla, Javier, and Eero P. Simoncelli. ["A parametric texture model based on joint statistics of complex wavelet coefficients."][2] _International journal of computer vision_ 40, no. 1 (2000): 49-70.

[1]: http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf
[2]: https://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf
