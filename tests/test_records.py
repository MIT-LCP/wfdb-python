import wfdb
import numpy as np

class test_rdsamp():

    # Test 1 - Format 212/Entire signal/Physical
    # Target file created with: rdsamp -r sampledata/100 -P | cut -f 2- >
    # target1
    def test_1(self):
        sig, fields = wfdb.srdsamp('sampledata/100')
        sig = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target1')
        assert np.array_equal(sig, targetsig)

    # Test 2 - Format 212/Selected Duration/Selected Channel/Digital.
    # Target file created with: rdsamp -r sampledata/100 -f 0.002 -t 30 -s 1 |
    # cut -f 2- > target2
    def test_2(self):
        sig = wfdb.srdsamp('sampledata/100', sampfrom=1,
                             sampto=10800, channels=[1], physical=0)
        targetsig = np.genfromtxt('tests/targetoutputdata/target2')
        targetsig = targetsig.reshape(len(targetsig), 1)
        assert np.array_equal(sig, targetsig)

    # Test 3 - Format 16/Entire signal/Digital
    # Target file created with: rdsamp -r sampledata/test01_00s | cut -f 2- >
    # target3
    def test_3(self):
        sig, fields = wfdb.srdsamp('sampledata/test01_00s', physical=0)
        targetsig = np.genfromtxt('tests/targetoutputdata/target3')
        assert np.array_equal(sig, targetsig)

    # Test 4 - Format 16 with byte offset/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/a103l -f 50 -t 160 -s 2 0
    # -P | cut -f 2- > target4
    def test_4(self):
        sig, fields = wfdb.srdsamp('sampledata/a103l',
                             sampfrom=12500, sampto=40000, channels=[2, 0])
        sig = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target4')
        assert np.array_equal(sig, targetsig)

    # Test 5 - Format 16 with byte offset/Selected Duration/Selected Channels/Digital
    # Target file created with: rdsamp -r sampledata/a103l -f 80 -s 0 1 | cut
    # -f 2- > target5
    def test_5(self):
        sig, fields = wfdb.srdsamp('sampledata/a103l',
                             sampfrom=20000, physical=0, channels=[0, 1])
        targetsig = np.genfromtxt('tests/targetoutputdata/target5')
        assert np.array_equal(sig, targetsig)

    # Test 6 - Format 80/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/3000003_0003 -f 1 -t 8 -s
    # 1 -P | cut -f 2- > target6
    def test_6(self):
        sig, fields = wfdb.srdsamp('sampledata/3000003_0003',
                             sampfrom=125, sampto=1000, channels=[1])
        sig = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target6')
        targetsig = targetsig.reshape(len(targetsig), 1)
        assert np.array_equal(sig, targetsig)

    # Test 7 - Multi-dat/Entire signal/Digital
    # Target file created with: rdsamp -r sampledata/s0010_re | cut -f 2- >
    # target7
    def test_7(self):
        sig, fields = wfdb.srdsamp('sampledata/s0010_re', physical=0)
        targetsig = np.genfromtxt('tests/targetoutputdata/target7')
        assert np.array_equal(sig, targetsig)

    # Test 8 - Multi-dat/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/s0010_re -f 5 -t 38 -P -s
    # 13 0 4 8 3 | cut -f 2- > target8
    def test_8(self):
        sig, fields = wfdb.srdsamp('sampledata/s0010_re', sampfrom=5000,
                             sampto=38000, channels=[13, 0, 4, 8, 3])
        sig = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target8')
        assert np.array_equal(sig, targetsig)

    # Test 9 - Format 12 multi-samples per frame and skew/Entire Signal/Digital
    # Target file created with: rdsamp -r sampledata/03700181 | cut -f 2- >
    # target9
    def test_9(self):
        sig, fields = wfdb.srdsamp('sampledata/03700181', physical=0)
        # The WFDB library rdsamp does not return the final N samples for all
        # channels due to the skew.
        sig = sig[:-4, :]
        # The WFDB python rdsamp does return the final N samples, filling in
        # NANs for end of skewed channels only.
        targetsig = np.genfromtxt('tests/targetoutputdata/target9')
        assert np.array_equal(sig, targetsig)

    # Test 10 - Format 12 multi-samples per frame and skew/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/03700181 -f 8 -t 128 -s 0
    # 2 -P | cut -f 2- > target10
    def test_10(self):
        sig, fields = wfdb.srdsamp('sampledata/03700181',
                             channels=[0, 2], sampfrom=1000, sampto=16000)
        sig = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target10')
        assert np.array_equal(sig, targetsig)

    #### Temporarily removing multi-segment tests due to differences in function workings

    # Test 11 - Multi-segment variable layout/Entire signal/Physical
    # Target file created with: rdsamp -r sampledata/matched/s25047/s25047-2704-05-04-10-44 -P | cut -f 2- > target11
    # def test_11(self):
        #sig, fields=rdsamp('sampledata/matched/s25047/s25047-2704-05-04-10-44')
        #sig=np.round(sig, decimals=8)
        # targetsig=np.genfromtxt('tests/targetoutputdata/target11')
        #assert np.array_equal(sig, targetsig)

    # Test 12 - Multi-segment variable layout/Selected duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/matched/s00001/s00001-2896-10-10-00-31 -f 70 -t 4000 -s 3 0 -P | cut -f 2- > target12
    # def test_12(self):
        #sig, fields=rdsamp('sampledata/matched/s00001/s00001-2896-10-10-00-31', sampfrom=8750, sampto=500000, channels=[3, 0])
        #sig=np.round(sig, decimals=8)
        # targetsig=np.genfromtxt('tests/targetoutputdata/target12')
        #assert np.array_equal(sig, targetsig)

    #################

    # Test 13 - Format 310/Selected Duration/Digital
    # Target file created with: rdsamp -r sampledata/3000003_0003 -f 0 -t 8.21 | cut -f 2- | wrsamp -o 310derive -O 310
    # rdsamp -r 310derive -f 0.007 | cut -f 2- > target13
    def test_13(self):
        sig, fields = wfdb.srdsamp('sampledata/310derive', sampfrom=2, physical=0)
        targetsig = np.genfromtxt('tests/targetoutputdata/target13')
        assert np.array_equal(sig, targetsig)

    # Test 14 - Format 311/Selected Duration/Physical
    # Target file created with: rdsamp -r sampledata/3000003_0003 -f 0 -t 8.21 -s 1 | cut -f 2- | wrsamp -o 311derive -O 311
    # rdsamp -r 311derive -f 0.005 -t 3.91 -P | cut -f 2- > target14
    def test_14(self):
        sig, fields = wfdb.srdsamp('sampledata/311derive', sampfrom=1, sampto=978)
        sig = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target14')
        targetsig = targetsig.reshape([977, 1])
        assert np.array_equal(sig, targetsig)
