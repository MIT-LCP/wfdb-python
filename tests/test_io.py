import wfdb
import numpy as np
import os


class test_record():
    # Target files created using the original WFDB Software Package
    # version 10.5.24

    # ----------------------- 1. Basic Tests -----------------------#

    # Format 16/Entire signal/Digital
    # Target file created with: rdsamp -r sample-data/test01_00s | cut -f 2- >
    # target1a
    def test_1a(self):
        record = wfdb.rdrecord('sample-data/test01_00s', physical=False)
        sig = record.d_signal
        targetsig = np.genfromtxt('tests/target-output/target1a')

        # Compare data streaming from physiobank
        pbrecord = wfdb.rdrecord('test01_00s', physical=False,
                                 pb_dir='macecgdb')

        # Test file writing
        record2 = wfdb.rdrecord('sample-data/test01_00s', physical=False)
        record2.sig_name = ['ECG_1', 'ECG_2', 'ECG_3', 'ECG_4']
        record2.wrsamp()
        recordwrite = wfdb.rdrecord('test01_00s', physical=False)

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(pbrecord)
        assert record2.__eq__(recordwrite)

    # Format 16 with byte offset/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sample-data/a103l -f 50 -t 160 -s 2 0
    # -P | cut -f 2- > target1b
    def test_1b(self):
        sig, fields = wfdb.rdsamp('sample-data/a103l',
                             sampfrom=12500, sampto=40000, channels=[2, 0])
        siground = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/target-output/target1b')

        # Compare data streaming from physiobank
        pbsig, pbfields = wfdb.rdsamp('a103l',
                                      pb_dir='challenge/2015/training',
                                      sampfrom=12500, sampto=40000,
                                      channels=[2, 0])
        assert np.array_equal(siground, targetsig)
        assert np.array_equal(sig, pbsig) and fields == pbfields

    # Format 16 with byte offset/Selected Duration/Selected Channels/Digital
    # Target file created with: rdsamp -r sample-data/a103l -f 80 -s 0 1 | cut
    # -f 2- > target1c
    def test_1c(self):
        record = wfdb.rdrecord('sample-data/a103l',
                               sampfrom=20000, channels=[0, 1], physical=False)
        sig = record.d_signal
        targetsig = np.genfromtxt('tests/target-output/target1c')

        # Compare data streaming from physiobank
        pbrecord = wfdb.rdrecord('a103l', pb_dir='challenge/2015/training',
                                 sampfrom=20000, channels=[0, 1],
                                 physical=False)

        # Test file writing
        record.wrsamp()
        recordwrite = wfdb.rdrecord('a103l', physical=False)

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(pbrecord)
        assert record.__eq__(recordwrite)

    # Format 80/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sample-data/3000003_0003 -f 1 -t 8 -s
    # 1 -P | cut -f 2- > target1d
    def test_1d(self):
        sig, fields = wfdb.rdsamp('sample-data/3000003_0003',
                             sampfrom=125, sampto=1000, channels=[1])
        siground = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/target-output/target1d')
        targetsig = targetsig.reshape(len(targetsig), 1)

        # Compare data streaming from physiobank
        pbsig, pbfields = wfdb.rdsamp('3000003_0003',
                                      pb_dir='mimic2wdb/30/3000003/',
                                      sampfrom=125, sampto=1000, channels=[1])

        assert np.array_equal(siground, targetsig)
        assert np.array_equal(sig, pbsig) and fields == pbfields


    # -------------------- 2. Special format tests ------------------ #

    # Format 212/Entire signal/Physical
    # Target file created with: rdsamp -r sample-data/100 -P | cut -f 2- >
    # target2a
    def test_2a(self):
        sig, fields = wfdb.rdsamp('sample-data/100')
        siground = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/target-output/target2a')

        # Compare data streaming from physiobank
        pbsig, pbfields = wfdb.rdsamp('100', pb_dir = 'mitdb')
        # This comment line was manually added and is not present in the
        # original physiobank record
        del(fields['comments'][0])

        assert np.array_equal(siground, targetsig)
        assert np.array_equal(sig, pbsig) and fields == pbfields

    # Format 212/Selected Duration/Selected Channel/Digital.
    # Target file created with: rdsamp -r sample-data/100 -f 0.002 -t 30 -s 1 |
    # cut -f 2- > target2b
    def test_2b(self):
        record = wfdb.rdrecord('sample-data/100', sampfrom=1,
                               sampto=10800, channels=[1], physical=False)
        sig = record.d_signal
        targetsig = np.genfromtxt('tests/target-output/target2b')
        targetsig = targetsig.reshape(len(targetsig), 1)

        # Compare data streaming from physiobank
        pbrecord = wfdb.rdrecord('100', sampfrom=1, sampto=10800, channels=[1],
                                 physical=False, pb_dir = 'mitdb')
        # This comment line was manually added and is not present in the
        # original physiobank record
        del(record.comments[0])

        # Test file writing
        record.wrsamp()
        recordwrite = wfdb.rdrecord('100', physical=False)

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(pbrecord)
        assert record.__eq__(recordwrite)


    # Format 212/Entire signal/Physical for odd sampled record
    # Target file created with: rdsamp -r sample-data/100_3chan -P | cut -f 2- >
    # target2c
    def test_2c(self):
        record = wfdb.rdrecord('sample-data/100_3chan')
        siground = np.round(record.p_signal, decimals=8)
        targetsig = np.genfromtxt('tests/target-output/target2c')

        # Test file writing
        record.d_signal = record.adc()
        record.wrsamp()
        recordwrite = wfdb.rdrecord('100_3chan')
        record.d_signal = None

        assert np.array_equal(siground, targetsig)
        assert record.__eq__(recordwrite)


    # Format 310/Selected Duration/Digital
    # Target file created with: rdsamp -r sample-data/3000003_0003 -f 0 -t 8.21 | cut -f 2- | wrsamp -o 310derive -O 310
    # rdsamp -r 310derive -f 0.007 | cut -f 2- > target2d
    def test_2d(self):
        record = wfdb.rdrecord('sample-data/310derive', sampfrom=2,
                               physical=False)
        sig = record.d_signal
        targetsig = np.genfromtxt('tests/target-output/target2d')
        assert np.array_equal(sig, targetsig)

    # Format 311/Selected Duration/Physical
    # Target file created with: rdsamp -r sample-data/3000003_0003 -f 0 -t 8.21 -s 1 | cut -f 2- | wrsamp -o 311derive -O 311
    # rdsamp -r 311derive -f 0.005 -t 3.91 -P | cut -f 2- > target2e
    def test_2e(self):
        sig, fields = wfdb.rdsamp('sample-data/311derive', sampfrom=1,
                                  sampto=978)
        sig = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/target-output/target2e')
        targetsig = targetsig.reshape([977, 1])
        assert np.array_equal(sig, targetsig)


    # ------------------- 3. Multi-dat file tests -------------------- #

    # Multi-dat/Entire signal/Digital
    # Target file created with: rdsamp -r sample-data/s0010_re | cut -f 2- >
    # target3a
    def test_3a(self):
        record= wfdb.rdrecord('sample-data/s0010_re', physical=False)
        sig = record.d_signal
        targetsig = np.genfromtxt('tests/target-output/target3a')

        # Compare data streaming from physiobank
        pbrecord= wfdb.rdrecord('s0010_re', physical=False,
                                pb_dir='ptbdb/patient001')

        # Test file writing
        record.wrsamp()
        recordwrite = wfdb.rdrecord('s0010_re', physical=False)

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(pbrecord)
        assert record.__eq__(recordwrite)

    # Multi-dat/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sample-data/s0010_re -f 5 -t 38 -P -s
    # 13 0 4 8 3 | cut -f 2- > target3b
    def test_3b(self):
        sig, fields = wfdb.rdsamp('sample-data/s0010_re', sampfrom=5000,
                             sampto=38000, channels=[13, 0, 4, 8, 3])
        siground = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/target-output/target3b')

        # Compare data streaming from physiobank
        pbsig, pbfields = wfdb.rdsamp('s0010_re', sampfrom=5000,
                                       pb_dir='ptbdb/patient001',
                                       sampto=38000, channels=[13, 0, 4, 8, 3])

        assert np.array_equal(siground, targetsig)
        assert np.array_equal(sig, pbsig) and fields == pbfields


    # ------------------- 4. Skew and multiple samples/frame tests ------------------- #

    # Format 16 multi-samples per frame and skew digital
    # Target file created with: rdsamp -r sample-data/test01_00s_skewframe | cut
    # -f 2- > target4a
    def test_4a(self):
        record = wfdb.rdrecord('sample-data/test01_00s_skewframe',
                               physical=False)
        sig = record.d_signal
        # The WFDB library rdsamp does not return the final N samples for all
        # channels due to the skew. The WFDB python rdsamp does return the final
        # N samples, filling in NANs for end of skewed channels only.
        sig = sig[:-3, :]

        targetsig = np.genfromtxt('tests/target-output/target4a')

        # Test file writing. Multiple samples per frame and skew.
        # Have to read all the samples in the record, ignoring skew
        recordnoskew = wfdb.rdrecord('sample-data/test01_00s_skewframe',
                                     physical=False,
                                     smooth_frames=False, ignore_skew=True)
        recordnoskew.wrsamp(expanded=True)
        # Read the written record
        recordwrite = wfdb.rdrecord('test01_00s_skewframe', physical=False)

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(recordwrite)

    # Format 12 multi-samples per frame and skew/Entire Signal/Digital
    # Target file created with: rdsamp -r sample-data/03700181 | cut -f 2- >
    # target4b
    def test_4b(self):
        record = wfdb.rdrecord('sample-data/03700181', physical=False)
        sig = record.d_signal
        # The WFDB library rdsamp does not return the final N samples for all
        # channels due to the skew.
        sig = sig[:-4, :]
        # The WFDB python rdsamp does return the final N samples, filling in
        # NANs for end of skewed channels only.
        targetsig = np.genfromtxt('tests/target-output/target4b')

        # Compare data streaming from physiobank
        pbrecord = wfdb.rdrecord('03700181', physical=False,
                                 pb_dir='mimicdb/037')

        # Test file writing. Multiple samples per frame and skew.
        # Have to read all the samples in the record, ignoring skew
        recordnoskew = wfdb.rdrecord('sample-data/03700181', physical=False,
                                     smooth_frames=False, ignore_skew=True)
        recordnoskew.wrsamp(expanded=True)
        # Read the written record
        recordwrite = wfdb.rdrecord('03700181', physical=False)

        assert np.array_equal(sig, targetsig)
        assert record.__eq__(pbrecord)
        assert record.__eq__(recordwrite)

    # Format 12 multi-samples per frame and skew/Selected Duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sample-data/03700181 -f 8 -t 128 -s 0
    # 2 -P | cut -f 2- > target4c
    def test_4c(self):
        sig, fields = wfdb.rdsamp('sample-data/03700181',
                             channels=[0, 2], sampfrom=1000, sampto=16000)
        siground = np.round(sig, decimals=8)
        targetsig = np.genfromtxt('tests/target-output/target4c')

        # Compare data streaming from physiobank
        pbsig, pbfields = wfdb.rdsamp('03700181', pb_dir = 'mimicdb/037',
                             channels=[0, 2], sampfrom=1000, sampto=16000)

        # Test file writing. Multiple samples per frame and skew.
        # Have to read all the samples in the record, ignoring skew
        recordnoskew = wfdb.rdrecord('sample-data/03700181', physical=False,
                                   smooth_frames=False, ignore_skew=True)
        recordnoskew.wrsamp(expanded=True)
        # Read the written record
        writesig, writefields = wfdb.rdsamp('03700181', channels=[0, 2],
                                              sampfrom=1000, sampto=16000)

        assert np.array_equal(siground, targetsig)
        assert np.array_equal(sig, pbsig) and fields == pbfields
        assert np.array_equal(sig, writesig) and fields == writefields


    # Format 16 multi-samples per frame and skew, read expanded signals
    # Target file created with: rdsamp -r sample-data/test01_00s_skewframe -P -H | cut
    # -f 2- > target4d
    def test_4d(self):
        record = wfdb.rdrecord('sample-data/test01_00s_skewframe',
                               smooth_frames=False)

        # Upsample the channels with lower samples/frame
        expandsig = np.zeros((7994, 3))
        expandsig[:,0] = np.repeat(record.e_p_signal[0][:-3],2)
        expandsig[:,1] = record.e_p_signal[1][:-6]
        expandsig[:,2] = np.repeat(record.e_p_signal[2][:-3],2)

        siground = np.round(expandsig, decimals=8)
        targetsig = np.genfromtxt('tests/target-output/target4d')

        assert np.array_equal(siground, targetsig)


    # ------------------------ 5. Multi-segment Tests ------------------------ #

    # Multi-segment variable layout/Selected duration. All samples contained in one segment.
    # Target file created with:
    # rdsamp -r sample-data/multi-segment/s00001/s00001-2896-10-10-00-31 -f s14428365 -t s14428375 -P | cut -f 2- > target5a
    def test_5a(self):
        record=wfdb.rdrecord('sample-data/multi-segment/s00001/s00001-2896-10-10-00-31',
                             sampfrom=14428365, sampto=14428375)
        siground=np.round(record.p_signal, decimals=8)
        targetsig=np.genfromtxt('tests/target-output/target5a')

        np.testing.assert_equal(siground, targetsig)

    # Multi-segment variable layout/Selected duration. Samples read from >1 segment
    # Target file created with:
    # rdsamp -r sample-data/multi-segment/s00001/s00001-2896-10-10-00-31 -f s14428364 -t s14428375 -P | cut -f 2- > target5b
    def test_5b(self):
        record=wfdb.rdrecord('sample-data/multi-segment/s00001/s00001-2896-10-10-00-31',
                             sampfrom=14428364, sampto=14428375)
        siground=np.round(record.p_signal, decimals=8)
        targetsig=np.genfromtxt('tests/target-output/target5b')

        np.testing.assert_equal(siground, targetsig)

    # Multi-segment fixed layout entire signal.
    # Target file created with: rdsamp -r sample-data/multi-segment/fixed1/v102s -P | cut -f 2- > target5c
    def test_5c(self):
        record=wfdb.rdrecord('sample-data/multi-segment/fixed1/v102s')
        siground=np.round(record.p_signal, decimals=8)
        targetsig=np.genfromtxt('tests/target-output/target5c')

        np.testing.assert_equal(siground, targetsig)

    # Multi-segment fixed layout/selected duration. All samples contained in one segment
    # Target file created with: rdsamp -r sample-data/multi-segment/fixed1/v102s -t s75000 -P | cut -f 2- > target5d
    def test_5d(self):
        record=wfdb.rdrecord('sample-data/multi-segment/fixed1/v102s',
                             sampto=75000)
        siground=np.round(record.p_signal, decimals=8)
        targetsig=np.genfromtxt('tests/target-output/target5d')

        np.testing.assert_equal(siground, targetsig)

    # Test 11 - Multi-segment variable layout/Entire signal/Physical
    # Target file created with: rdsamp -r sample-data/matched/s25047/s25047-2704-05-04-10-44 -P | cut -f 2- > target11
    # def test_11(self):
        #sig, fields=rdsamp('sample-data/matched/s25047/s25047-2704-05-04-10-44')
        #sig=np.round(sig, decimals=8)
        # targetsig=np.genfromtxt('tests/target-output/target11')
        #assert np.array_equal(sig, targetsig)

    # Test 12 - Multi-segment variable layout/Selected duration/Selected Channels/Physical
    # Target file created with: rdsamp -r sample-data/matched/s00001/s00001-2896-10-10-00-31 -f s -t 4000 -s 3 0 -P | cut -f 2- > target12
    #def test_12(self):
    #    record=rdsamp('sample-data/matched/s00001/s00001-2896-10-10-00-31', sampfrom=8750, sampto=500000)
    #    siground=np.round(record.p_signal, decimals=8)
    #    targetsig=np.genfromtxt('tests/target-output/target12')
    #
    #    assert np.array_equal(sig, targetsig)


    # Cleanup written files
    @classmethod
    def tearDownClass(self):

        writefiles = ['03700181.dat','03700181.hea','100.atr','100.dat',
                      '100.hea','1003.atr','100_3chan.dat','100_3chan.hea',
                      '12726.anI','a103l.hea','a103l.mat','s0010_re.dat',
                      's0010_re.hea','s0010_re.xyz','test01_00s.dat',
                      'test01_00s.hea','test01_00s_skewframe.hea']

        for file in writefiles:
            if os.path.isfile(file):
                os.remove(file)
