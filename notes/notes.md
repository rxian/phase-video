Algorithm Overview ([1], Fig. 2)
- Compute the complex steerable pyramid for each frame of the video (processed on (x,y)-planes, where the video sequence lies on (x,y,t)-hyperplane; now the changes in the phase component over time corresponds to motion).
- Perform bandpass temporal filtering on the phase component of each image in the pyramid to isolate motion at specific frequencies (processed on t-axis, and now the amplitude of the resulting "image" corresponds to the amount of motion).
- Multiply the resulting "images" by $\alpha$, and add it back to the phase component of the respective frames in the pyramid (positive coefficient gives motion amplification, and negative gives attenuation; some temporal interpolation could be done between time frames?).
- Collapse the pyramid.

[1]: phase-video.pdf
