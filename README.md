# wfdb-python

[![Build Status](https://travis-ci.org/MIT-LCP/wfdb-python.svg?branch=master)](https://travis-ci.org/MIT-LCP/wfdb-python)

## Introduction
Native python scripts for reading and writing WFDB signals and annotations. Currently only contains reading signals. 


## Usage

rdsamp - Read a WFDB file and return the signal as a numpy array and the metadata as a dictionary. 

```
sig, fields = rdsamp(recordname, sampfrom, sampto, channels, physical, stacksegments) 
```

Mandatory Argument: 
recordname - The name of the WFDB record to be read (without any file extensions).

Optional Arguments:
sampfrom (default=0)- The starting sample number to read for each channel.
sampto (default=length of entire signal)- The final sample number to read for each channel.
channels (default=all channels) - The channel indices to be returned.
physical (default=1) - Flag (0 or 1) that specifies whether to return signals in physical or digital units. 
stacksegments (default 1) - Flag (0 or 1) used only for multi-segment files. Specifies whether to return the signal as a single stacked/concatenated numpy array (1) or as a list of one numpy array for each segment (0). For empty segments, stacked format will fill in Nan values, and non-stacked format will fill in a single integer specifying the empty segment length. The corresponding field item will not be a dictionary but rather a single string: 'Empty Segment'.  


