import numpy as np
from wfdb import readsignal
        
class test_rdsamp():
        
    # Test 1 - Format 212/Entire signal/Physical 
    # Target file created with: rdsamp -r sampledata/100 -P | cut -f 2- > target1      
    def test_1(self):
        sig, fields=readsignal.rdsamp('sampledata/100')
        sig=np.round(sig, decimals=8)
        targetsig=np.genfromtxt('tests/targetoutputdata/target1')
        assert np.array_equal(sig, targetsig)
        
    # Test 2 - Format 212/Selected Duration/Selected Channel/Digital
    # Target file created with: rdsamp -r sampledata/100 -f 1 -t 30 -s 1 | cut -f 2- > target2
    def test_2(self):
        sig, fields=readsignal.rdsamp('sampledata/100', sampfrom=360, sampto=10800, channels=[1], physical=0) 
        targetsig=np.genfromtxt('tests/targetoutputdata/target2')
        targetsig=targetsig.reshape(len(targetsig), 1)
        assert np.array_equal(sig, targetsig)
    
    # Test 3 - Format 16/Entire signal/Digital
    # Target file created with: rdsamp -r sampledata/test01_00s | cut -f 2- > target3
    def test_3(self):
        sig, fields=readsignal.rdsamp('sampledata/test01_00s', physical=0)
        targetsig=np.genfromtxt('tests/targetoutputdata/target3')
        assert np.array_equal(sig, targetsig)
        
    # Test 4 - Format 16 with byte offset/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/a103l -f 50 -t 160 -s 2 0 -P | cut -f 2- > target4
    def test_4(self):
        sig, fields=readsignal.rdsamp('sampledata/a103l', sampfrom=12500, sampto=40000, channels=[2, 0])
        sig=np.round(sig, decimals=8)
        targetsig=np.genfromtxt('tests/targetoutputdata/target4')
        assert np.array_equal(sig, targetsig) 
    
    # Test 5 - Format 16 with byte offset/Selected Duration/Selected Channels/Digital
    # Target file created with: rdsamp -r sampledata/a103l -f 80 -s 0 1 | cut -f 2- > target5
    def test_5(self):
        sig, fields=readsignal.rdsamp('sampledata/a103l', sampfrom=20000, physical=0, channels=[0, 1])
        targetsig=np.genfromtxt('tests/targetoutputdata/target5')
        assert np.array_equal(sig, targetsig) 
    
    # Test 6 - Format 80/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/3000003_0003 -f 1 -t 8 -s 1 -P | cut -f 2- > target6
    def test_6(self):
        sig, fields=readsignal.rdsamp('sampledata/3000003_0003', sampfrom=125, sampto=1000, channels=[1])
        sig=np.round(sig, decimals=8)
        targetsig=np.genfromtxt('tests/targetoutputdata/target6')
        targetsig=targetsig.reshape(len(targetsig), 1)
        assert np.array_equal(sig, targetsig) 
    
    # Test 7 - Multi-dat/Entire signal/Digital
    # Target file created with: rdsamp -r sampledata/s0010_re | cut -f 2- > target7
    def test_7(self):
        sig, fields=readsignal.rdsamp('sampledata/s0010_re', physical=0)
        targetsig=np.genfromtxt('tests/targetoutputdata/target7')
        assert np.array_equal(sig, targetsig) 
    
    # Test 8 - Multi-dat/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/s0010_re -f 5 -t 38 -P -s 13 0 4 8 3 | cut -f 2- > target8
    def test_8(self):
        sig, fields=readsignal.rdsamp('sampledata/s0010_re', sampfrom=5000, sampto=38000, channels=[13, 0, 4, 8, 3])
        sig=np.round(sig, decimals=8)
        targetsig=np.genfromtxt('tests/targetoutputdata/target8')
        assert np.array_equal(sig, targetsig) 
    
    # Test 9 - Format 12 multi-samples/frame and skew/Entire Signal/Digital
    # Target file created with: rdsamp -r sampledata/03700181 | cut -f 2- > target9
    def test_9(self):
        sig, fields=readsignal.rdsamp('sampledata/03700181', physical=0)
        sig=sig[:-4,:] # The WFDB library rdsamp does not return the final N samples for all channels due to the skew. 
        # The WFDB python rdsamp does return the final N samples, filling in NANs for skewed channels only. 
        sig=np.round(sig, decimals=8)
        targetsig=np.genfromtxt('tests/targetoutputdata/target9')
        assert np.array_equal(sig, targetsig)
        