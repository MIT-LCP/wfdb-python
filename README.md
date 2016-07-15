# wfdb-python

[![Build Status](https://travis-ci.org/MIT-LCP/wfdb-python.svg?branch=master)](https://travis-ci.org/MIT-LCP/wfdb-python)

## Introduction
Native python scripts for reading and writing WFDB signals and annotations. Currently only contains reading signals. 


## Usage

<strong>rdsamp</strong> - Read a WFDB file and return the signal as a numpy array and the metadata as a dictionary. 

```
from WFDB import readsignal
sig, fields = readsignal.rdsamp(recordname, sampfrom, sampto, channels, physical, stacksegments) 
```

Input Arguments: 
<ul>
<li><code>recordname</code> (mandatory)- The name of the WFDB record to be read (without any file extensions).</li>
<li>sampfrom (default=0)- The starting sample number to read for each channel.</li>
<li>sampto (default=length of entire signal)- The final sample number to read for each channel.</li>
<li>channels (default=all channels) - The channel indices to be returned.</li>
<li>physical (default=1) - Flag (0 or 1) that specifies whether to return signals in physical or digital units. </li>
<li>stacksegments (default 1) - Flag (0 or 1) used only for multi-segment files. Specifies whether to return the signal as a single stacked/concatenated numpy array (1) or as a list of one numpy array for each segment (0). </li>
</ul>

Output Arguments:
<ul>
	<li>sig - An nxm numpy array where n is the signal length and m is the number of channels. <br>If the input record is a multi-segment record, depending on the input stacksegments flag, sig will either be a single stacked/concatenated numpy array (1) or a list of one numpy array for each segment (0). For empty segments, stacked format will fill in Nan values, and non-stacked format will fill in a single integer specifying the empty segment length.</li>
	<li>fields - A dictionary of metadata about the record extracted or deduced from the header/signal file. <br>If the input record is a multi-segment record, the output argument will be a list of dictionaries:
	<ul>
		<li>The first list element will be a dictionary of metadata about the master header.</li> 
		<li>If the record is in variable layout format, the next list element will be a dictionary of metadata about the layout specification header.</li>
		<li>The last list element will be a list of dictionaries of metadata for each segment. For empty segments, the dictionary will be replaced by a single string: 'Empty Segment'</li>
	</ul>
</ul>


## Based on the original WFDB software package specifications

[WFDB Software Package](http://physionet.org/physiotools/wfdb.shtml) 
<br>[WFDB Applications Guide](http://physionet.org/physiotools/wag/) 
<br>[WFDB Header File Specifications](https://physionet.org/physiotools/wag/header-5.htm)
