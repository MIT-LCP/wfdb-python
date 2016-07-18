import numpy as np
from wfdb import readsignal

# NB: write np.array to CSV with:
# sig[:,1].tofile('test01_00s_channel1.csv',sep=',',format='%s')

#class test_read_format_16(): 

    #def setUp(self):
        # load the sample binary file
        #self.sig, self.fields = readsignal.rdsamp('sampledata/test01_00s', physical=0) 

    #def test_channel1(self):
        # load the expected results
        #channel1 = np.genfromtxt('tests/targetoutputdata/test01_00s_channel1.csv', delimiter=',')
        #assert np.array_equal(self.sig[:,0],channel1)

    #def test_channel2(self):
        # load the expected results
        #channel2 = np.genfromtxt('tests/targetoutputdata/test01_00s_channel2.csv', delimiter=',')
        #assert np.array_equal(self.sig[:,1],channel2)

    #def test_channel3(self):
        # load the expected results
        #channel3 = np.genfromtxt('tests/targetoutputdata/test01_00s_channel3.csv', delimiter=',')
        #assert np.array_equal(self.sig[:,2],channel3)
    
    #def test_channel4(self):
        # load the expected results
        #channel4 = np.genfromtxt('tests/targetoutputdata/test01_00s_channel4.csv', delimiter=',')
        #assert np.array_equal(self.sig[:,3],channel4)

        
class testrdsamp():
    
    def setUp(self):
        sig1, fields1 = readsignal.rdsamp('sampledata/100')
        sig2, fields2=readsignal.rdsamp('sampledata/100', sampfrom=360, sampto=10800, channels=[1], physical=0) 
        sig2, fields2=readsignal.rdsamp('sampledata/test01_00s', physical=0)
        
    # Test 1 - Format 212/Entire signal/Physical 
    # Target file created with: rdsamp -r sampledata/100 -P | cut -f 2- > target1      
    def runtest1(self):
        targetsig1=np.genfromtxt('tests/targetoutputdata/target1')
        assert np.array_equal(sig1, targetsig1)
        
    # Test 2 - Format 212/Selected Duration/Selected Channel/Digital
    # Target file created with: rdsamp -r sampledata/100 -f 1 -t 30 -s 1 | cut -f 2- > target2
    def runtest2(self):
        targetsig2=np.genfromtxt('tests/targetoutputdata/target2')
        assert np.array_equal(sig2, targetsig2)
    
    # Test 3 - Format 16/Entire signal/Digital
    # Target file created with: rdsamp -r sampledata/test01_00s | cut -f 2- > target3
    def runtest3(self):
        targetsig3=np.genfromtxt('tests/targetoutputdata/target3')
        assert np.array_equal(sig3, targetsig3)