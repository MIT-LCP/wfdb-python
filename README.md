# wfdb-python

[![Build Status](https://travis-ci.org/MIT-LCP/wfdb-python.svg?branch=master)](https://travis-ci.org/MIT-LCP/wfdb-python)

<p align="center" >
  <img src="https://raw.githubusercontent.com/MIT-LCP/wfdb-python/master/demoimg1.png" alt="wfdbsignals" title="wfdbsignals"/>
</p>

## Introduction
<p>Native python scripts for reading and writing WFDB signals and annotations. Currently in development. Library files are in the <strong>wfdb</strong> directory.</p> 

<ul>
	<li>15 July 2016 - <code>rdsamp</code> (for reading WFDB signals) is ready for beta usage.</li>
	<li>27 July 2016 - <code>rdann</code> (for reading WFDB annotations) is ready for beta usage.</li>
	<li>27 July 2016 - <code>plotsigs</code> is able to plot signals and annotations output by <em>rdsamp</em> and <em>rdann</em></li>
</ul>



## Usage

See the <strong>wfdbdemo.ipynb</strong> file for example scripts on how to call the functions.

### Reading Signals

<strong>rdsamp</strong> - Read a WFDB file and return the signal as a numpy array and the metadata as a dictionary. 

```
import numpy as np
from wfdb import readsignal
sig, fields = readsignal.rdsamp(recordname, sampfrom, sampto, channels, physical, stacksegments) 
```

Input Arguments: 
<ul>
<li><code>recordname</code> (mandatory) - The name of the WFDB record to be read (without any file extensions).</li>
<li><code>sampfrom</code> (default=0) - The starting sample number to read for each channel.</li>
<li><code>sampto</code> (default=length of entire signal)- The final sample number to read for each channel.</li>
<li><code>channels</code> (default=all channels) - Indices specifying the channel to be returned.</li>
<li><code>physical</code> (default=1) - Flag that specifies whether to return signals in physical (1) or digital (0) units.</li>
<li><code>stacksegments</code> (default=1) - Flag used only for multi-segment files. Specifies whether to return the signal as a single stacked/concatenated numpy array (1) or as a list of one numpy array for each segment (0). </li>
</ul>

Output Arguments:
<ul>
	<li><code>sig</code> - An nxm numpy array where n is the signal length and m is the number of channels. <br>If the input record is a multi-segment record, depending on the input stacksegments flag, sig will either be a single stacked/concatenated numpy array (1) or a list of one numpy array for each segment (0). For empty segments, stacked format will contain Nan values, and non-stacked format will contain a single integer specifying the length of the empty segment.</li>
	<li><code>fields</code> - A dictionary of metadata about the record extracted or deduced from the header/signal file. <br>If the input record is a multi-segment record, the output argument will be a list of dictionaries:
	<ul>
		<li>The first list element will be a dictionary of metadata about the master header.</li> 
		<li>If the record is in variable layout format, the next list element will be a dictionary of metadata about the layout specification header.</li>
		<li>The last list element will be a list of dictionaries of metadata for each segment. For empty segments, the dictionary will be replaced by a single string: 'Empty Segment'</li>
	</ul>
</ul>


### Reading Annotations

<strong>rdann</strong> - Read a WFDB annotation file <code>recordname.annot</code> and return the fields as lists or arrays.

```
import numpy as np
from wfdb import readannot
annsamp, anntype, num, subtype, chan, aux, annfs) = readannot.rdann(recordname, annot, sampfrom, sampto, anndisp)
```

Input Arguments: 
<ul>
<li><code>recordname</code> (required) - The record name of the WFDB annotation file. ie. for file <code>100.atr</code> recordname='100'.</li>
<li><code>annot</code> (required) - The annotator extension of the annotation file. ie. for file <code>100.atr</code> annot='atr'.</li>
<li><code>sampfrom</code> (default=0)- The minimum sample number for annotations to be returned.</li>
<li><code>sampto</code> (default=the final annotation sample) - The maximum sample number for annotations to be returned.</li>
<li><code>anndisp</code> (default=1) - The annotation display flag that controls the data type of the <code>anntype</code> output parameter. <code>anntype</code> will either be an integer key(0), a shorthand display symbol(1), or a longer annotation code.</li>
</ul>

Output arguments: 

<ul>
<li><code>annsamp</code> - The annotation location in samples relative to the beginning of the record.</li>
<li><code>anntype</code> - The annotation type according the the standard WFDB keys.</li>
<li><code>num</code> - The marked annotation number. This is not equal to the index of the current annotation.</li>
<li><code>subtype</code> - The marked class/category of the annotation.</li>
<li><code>aux</code> - The auxiliary information string for the annotation.</li>
<li><code>annfs</code> - The sampling frequency written in the beginning of the annotation file if present.</li>
</ul>

<strong>*NOTE</strong>: Every annotation contains the 'annsamp' and 'anntype' field. All other fields default to 0 or empty if not present. 


### Plotting Data

<strong>plotsigs</strong> - Subplot and label each channel of an nxm signal on a graph. Also subplot annotation locations on top of selected channels if present.  

```
import numpy as np
from wfdb import readsignal
from wfdb import plotwfdb
sig, fields = readsignal.rdsamp(recordname)
plotwfdb.plotsigs(sig, fields, annsamp, annch, title, plottime): 
 
```

Input Arguments: 
<ul>
	<li>sig (required)- An nxm numpy array where n is the number of samples and m is the number of channels. Standard first output of the <code>rdsamp</code> function.</li>
	<li>fields (required) - A dictionary of metadata about the signal. Standard second output of the <code>rdsamp</code> function.</li>
	<li>annsamp (optional) - An 1d numpy array that contains annotation sample locations. Standard first output of the <code>rdann</code> function. 
	<li>annch</li> (default=[0]) - A list of channels on which to plot the annotations.  
	<li>title (optional)- A string containing the title of the graph.</li>
	<li>plottime (default=1) - Flag that specifies whether to plot the x axis as time (1) or samples (0). Defaults to samples if the input <code>fields</code> dictionary does not contain a value for <code>fs</code>.</li>
</ul>



## Based on the original WFDB software package specifications

[WFDB Software Package](http://physionet.org/physiotools/wfdb.shtml) 
<br>[WFDB Applications Guide](http://physionet.org/physiotools/wag/) 
<br>[WFDB Header File Specifications](https://physionet.org/physiotools/wag/header-5.htm)
