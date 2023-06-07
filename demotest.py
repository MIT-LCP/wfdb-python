from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath

import wfdb
# Demo 1 - Read a WFDB record using the 'rdrecord' function into a wfdb.Record object.
# Plot the signals, and show the data.
record = wfdb.rdrecord('sample-data/03700181')
wfdb.plot_wfdb(record=record, title='Record 03700181 from PhysioNet Challenge 2015')
display(record.__dict__)

# Can also read the same files hosted on PhysioNet https://physionet.org/content/challenge-2015/1.0.0
# in the /training/ database subdirectory.
record2 = wfdb.rdrecord('a103l', pn_dir='challenge-2015/training/')
signals, fields = wfdb.rdsamp('sample-data/s0010_re', channels=[14, 0, 5, 10], sampfrom=100, sampto=15000)
display(signals)
display(fields)

# Can also read the same files hosted on Physionet
signals2, fields2 = wfdb.rdsamp('s0010_re', channels=[14, 0, 5, 10], sampfrom=100, sampto=15000, pn_dir='ptbdb/patient001/')

record = wfdb.rdheader('sample-data/drive02')
display(record.__dict__)

# Can also read the same file hosted on Physionet
record2 = wfdb.rdheader('drive02', pn_dir='drivedb')

# Demo 4 - Read part of a WFDB annotation file into a wfdb.Annotation object, and plot the samples
annotation = wfdb.rdann('sample-data/100', 'atr', sampfrom=100000, sampto=110000)
annotation.fs = 360
wfdb.plot_wfdb(annotation=annotation, time_units='minutes')

# Can also read the same file hosted on PhysioNet
annotation2 = wfdb.rdann('100', 'atr', sampfrom=100000, sampto=110000, pn_dir='mitdb')

# Demo 5 - Read a WFDB record and annotation. Plot all channels, and the annotation on top of channel 0.
record = wfdb.rdrecord('sample-data/100', sampto = 15000)
annotation = wfdb.rdann('sample-data/100', 'atr', sampto = 15000)

wfdb.plot_wfdb(record=record, annotation=annotation,
               title='Record 100 from MIT-BIH Arrhythmia Database',
               time_units='seconds')
record = wfdb.rdrecord('sample-data/multi-segment/p000878/3269321_0001')
wfdb.plot_wfdb(record=record, title='Record p000878/3269321_0001')
display(record.__dict__)

# Can also read the same files hosted on PhysioNet (takes long to stream the many large files)
signals, fields = wfdb.rdsamp('3269321_0001', pn_dir = 'mimic3wdb/matched/p00/p000878')
wfdb.plot_items(signal=signals, fs=fields['fs'], title='Record p000878/3269321_0001')
display((signals, fields))