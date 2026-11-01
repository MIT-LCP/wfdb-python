**Fix smoothing of frames with invalid samples**
  When a signal with more than one sample per frame is read with ``smooth_frames=True`` (the default), a frame that contains an invalid sample now gives an invalid value (NaN in physical units), as in the WFDB library. The invalid sample value used to be averaged with the other samples, which gave a large but valid-looking value.
