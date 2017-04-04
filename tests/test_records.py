import wfdb
import numpy as np

# Target files created using the original WFDB Software Package version 10.5.24
class test_rdsamp():

    # Test 1 - Format 212/Entire signal/Physical
    # Target file created with: rdsamp -r sampledata/100 -P | cut -f 2- >
    # target1
    def test_1(self):
        sig, fields = wfdb.srdsamp('sampledata/100')
        siground = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target1')

        # Compare data streaming from physiobank
        pbsig, pbfields = wfdb.srdsamp('100', pbdir = 'mitdb')
        # This comment line was manually added and is not present in the original physiobank record
        del(fields['comments'][0])

        # Test file writing
        assert np.array_equal(siground, targetsig)
        assert np.array_equal(sig, pbsig) and fields == pbfields

    # Test 2 - Format 212/Selected Duration/Selected Channel/Digital.
    # Target file created with: rdsamp -r sampledata/100 -f 0.002 -t 30 -s 1 |
    # cut -f 2- > target2
    def test_2(self):
        record = wfdb.rdsamp('sampledata/100', sampfrom=1,
                             sampto=10800, channels=[1], physical=False)
        sig = record.d_signals
        targetsig = np.genfromtxt('tests/targetoutputdata/target2')
        targetsig = targetsig.reshape(len(targetsig), 1)

        # Compare data streaming from physiobank
        pbrecord = wfdb.rdsamp('100', sampfrom=1, sampto=10800, channels=[1], physical=False, pbdir = 'mitdb')
        # This comment line was manually added and is not present in the original physiobank record
        del(record.comments[0])

        # Test file writing
        record.wrsamp()
        recordwrite = wfdb.rdsamp('100')

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(pbrecord)
        assert record.__eq__(recordwrite)

    # Test 3 - Format 16/Entire signal/Digital
    # Target file created with: rdsamp -r sampledata/test01_00s | cut -f 2- >
    # target3
    def test_3(self):
        record = wfdb.rdsamp('sampledata/test01_00s', physical=False)
        sig = record.d_signals
        targetsig = np.genfromtxt('tests/targetoutputdata/target3')

        # Compare data streaming from physiobank
        pbrecord = wfdb.rdsamp('test01_00s', physical=False, pbdir = 'macecgdb')

        # Test file writing
        record2 = wfdb.rdsamp('sampledata/test01_00s', physical=False)
        record2.signame = ['ECG1', 'ECG2', 'ECG3', 'ECG4']
        record2.wrsamp()
        recordwrite = wfdb.rdsamp('test01_00s', physical=False)

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(pbrecord)
        assert record2.__eq__(recordwrite)

    # Test 4 - Format 16 with byte offset/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/a103l -f 50 -t 160 -s 2 0
    # -P | cut -f 2- > target4
    def test_4(self):
        sig, fields = wfdb.srdsamp('sampledata/a103l',
                             sampfrom=12500, sampto=40000, channels=[2, 0])
        siground = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target4')

        # Compare data streaming from physiobank
        pbsig, pbfields = wfdb.srdsamp('a103l', pbdir = 'challenge/2015/training',
                             sampfrom=12500, sampto=40000, channels=[2, 0])
        assert np.array_equal(siground, targetsig)
        assert np.array_equal(sig, pbsig) and fields == pbfields

    # Test 5 - Format 16 with byte offset/Selected Duration/Selected Channels/Digital
    # Target file created with: rdsamp -r sampledata/a103l -f 80 -s 0 1 | cut
    # -f 2- > target5
    def test_5(self):
        record = wfdb.rdsamp('sampledata/a103l',
                             sampfrom=20000, channels=[0, 1], physical=False)
        sig = record.d_signals
        targetsig = np.genfromtxt('tests/targetoutputdata/target5')

        # Compare data streaming from physiobank
        pbrecord = wfdb.rdsamp('a103l', pbdir = 'challenge/2015/training',
                             sampfrom=20000, channels=[0, 1], physical=False)

        # Test file writing
        record.wrsamp()
        recordwrite = wfdb.rdsamp('a103l')

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(pbrecord)
        assert record.__eq__(recordwrite)

    # Test 6 - Format 80/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/3000003_0003 -f 1 -t 8 -s
    # 1 -P | cut -f 2- > target6
    def test_6(self):
        sig, fields = wfdb.srdsamp('sampledata/3000003_0003',
                             sampfrom=125, sampto=1000, channels=[1])
        siground = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target6')
        targetsig = targetsig.reshape(len(targetsig), 1)

        # Compare data streaming from physiobank
        pbsig, pbfields = wfdb.srdsamp('3000003_0003', pbdir = 'mimic2wdb/30/3000003/',
                             sampfrom=125, sampto=1000, channels=[1])

        assert np.array_equal(siground, targetsig)
        assert np.array_equal(sig, pbsig) and fields == pbfields

    # Test 7 - Multi-dat/Entire signal/Digital
    # Target file created with: rdsamp -r sampledata/s0010_re | cut -f 2- >
    # target7
    def test_7(self):
        record= wfdb.rdsamp('sampledata/s0010_re', physical=False)
        sig = record.d_signals
        targetsig = np.genfromtxt('tests/targetoutputdata/target7')

        # Compare data streaming from physiobank
        pbrecord= wfdb.rdsamp('s0010_re', physical=False, pbdir = 'ptbdb/patient001')

        # Test file writing
        record.wrsamp()
        recordwrite = wfdb.rdsamp('s0010_re', physical=False)

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(pbrecord)
        assert record.__eq__(recordwrite)

    # Test 8 - Multi-dat/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/s0010_re -f 5 -t 38 -P -s
    # 13 0 4 8 3 | cut -f 2- > target8
    def test_8(self):
        sig, fields = wfdb.srdsamp('sampledata/s0010_re', sampfrom=5000,
                             sampto=38000, channels=[13, 0, 4, 8, 3])
        siground = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target8')

        # Compare data streaming from physiobank
        pbsig, pbfields = wfdb.srdsamp('s0010_re', sampfrom=5000, pbdir = 'ptbdb/patient001',
                             sampto=38000, channels=[13, 0, 4, 8, 3])

        assert np.array_equal(siground, targetsig)
        assert np.array_equal(sig, pbsig) and fields == pbfields

    # Test 9 - Format 12 multi-samples per frame and skew/Entire Signal/Digital
    # Target file created with: rdsamp -r sampledata/03700181 | cut -f 2- >
    # target9
    def test_9(self):
        record = wfdb.rdsamp('sampledata/03700181', physical=False)
        sig = record.d_signals
        # The WFDB library rdsamp does not return the final N samples for all
        # channels due to the skew.
        sig = sig[:-4, :]
        # The WFDB python rdsamp does return the final N samples, filling in
        # NANs for end of skewed channels only.
        targetsig = np.genfromtxt('tests/targetoutputdata/target9')

        # Compare data streaming from physiobank
        pbrecord = wfdb.rdsamp('03700181', physical=False, pbdir = 'mimicdb/037')

        # Test file writing. Multiple Samples per frame. To do...
        #record.wrsamp()
        #recordwrite = wfdb.rdsamp('03700181', physical=False)

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(pbrecord)
        #assert record.__eq__(recordwrite)

    # Test 10 - Format 12 multi-samples per frame and skew/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sampledata/03700181 -f 8 -t 128 -s 0
    # 2 -P | cut -f 2- > target10
    def test_10(self):
        sig, fields = wfdb.srdsamp('sampledata/03700181',
                             channels=[0, 2], sampfrom=1000, sampto=16000)
        siground = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/targetoutputdata/target10')

        # Compare data streaming from physiobank
        pbsig, pbfields = wfdb.srdsamp('03700181', pbdir = 'mimicdb/037',
                             channels=[0, 2], sampfrom=1000, sampto=16000)

        assert np.array_equal(siground, targetsig)
        assert np.array_equal(sig, pbsig) and fields == pbfields

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
        record = wfdb.rdsamp('sampledata/310derive', sampfrom=2, physical=False)
        sig = record.d_signals
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


