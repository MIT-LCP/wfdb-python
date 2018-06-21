import os
import shutil

import numpy as np

import wfdb


class TestRecord():
    """
    Test read and write of single segment wfdb records, including
    Physionet streaming.

    Target files created using the original WFDB Software Package
    version 10.5.24

    """

    # ----------------------- 1. Basic Tests -----------------------#

    def test_1a(self):
        """
        Format 16, entire signal, digital.

        Target file created with:
            rdsamp -r sample-data/test01_00s | cut -f 2- > record-1a
        """
        record = wfdb.rdrecord('sample-data/test01_00s', physical=False)
        sig = record.d_signal
        sig_target = np.genfromtxt('tests/target-output/record-1a')

        # Compare data streaming from physiobank
        record_pb = wfdb.rdrecord('test01_00s', physical=False,
                                  pb_dir='macecgdb')

        # Test file writing
        record_2 = wfdb.rdrecord('sample-data/test01_00s', physical=False)
        record_2.sig_name = ['ECG_1', 'ECG_2', 'ECG_3', 'ECG_4']
        record_2.wrsamp()
        record_write = wfdb.rdrecord('test01_00s', physical=False)

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pb)
        assert record_2.__eq__(record_write)

    def test_1b(self):
        """
        Format 16, byte offset, selected duration, selected channels,
        physical.

        Target file created with:
            rdsamp -r sample-data/a103l -f 50 -t 160 -s 2 0 -P | cut -f 2- > record-1b
        """
        sig, fields = wfdb.rdsamp('sample-data/a103l', sampfrom=12500,
                                  sampto=40000, channels=[2, 0])
        sig_round = np.round(sig, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-1b')

        # Compare data streaming from physiobank
        sig_pb, fields_pb = wfdb.rdsamp('a103l',
                                        pb_dir='challenge/2015/training',
                                        sampfrom=12500, sampto=40000,
                                        channels=[2, 0])

        # Option of selecting channels by name
        sig_named, fields_named = wfdb.rdsamp('sample-data/a103l',
                                              sampfrom=12500, sampto=40000,
                                              channel_names=['PLETH', 'II'])

        assert np.array_equal(sig_round, sig_target)
        assert np.array_equal(sig, sig_pb) and fields == fields_pb
        assert np.array_equal(sig, sig_named) and fields == fields_named

    def test_1c(self):
        """
        Format 16, byte offset, selected duration, selected channels,
        digital.

        Target file created with:
            rdsamp -r sample-data/a103l -f 80 -s 0 1 | cut -f 2- > record-1c
        """
        record = wfdb.rdrecord('sample-data/a103l',
                               sampfrom=20000, channels=[0, 1], physical=False)
        sig = record.d_signal
        sig_target = np.genfromtxt('tests/target-output/record-1c')

        # Compare data streaming from physiobank
        record_pb = wfdb.rdrecord('a103l', pb_dir='challenge/2015/training',
                                  sampfrom=20000, channels=[0, 1],
                                  physical=False)

        # Test file writing
        record.wrsamp()
        record_write = wfdb.rdrecord('a103l', physical=False)

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pb)
        assert record.__eq__(record_write)

    def test_1d(self):
        """
        Format 80, selected duration, selected channels, physical

        Target file created with:
            rdsamp -r sample-data/3000003_0003 -f 1 -t 8 -s 1 -P | cut -f 2- > record-1d
        """
        sig, fields = wfdb.rdsamp('sample-data/3000003_0003', sampfrom=125,
                                  sampto=1000, channels=[1])
        sig_round = np.round(sig, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-1d')
        sig_target = sig_target.reshape(len(sig_target), 1)

        # Compare data streaming from physiobank
        sig_pb, fields_pb = wfdb.rdsamp('3000003_0003',
                                        pb_dir='mimic2wdb/30/3000003/',
                                        sampfrom=125, sampto=1000,
                                        channels=[1])

        assert np.array_equal(sig_round, sig_target)
        assert np.array_equal(sig, sig_pb) and fields == fields_pb


    # ------------------ 2. Special format records ------------------ #

    def test_2a(self):
        """
        Format 212, entire signal, physical.

        Target file created with:
            rdsamp -r sample-data/100 -P | cut -f 2- > record-2a
        """
        sig, fields = wfdb.rdsamp('sample-data/100')
        sig_round = np.round(sig, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-2a')

        # Compare data streaming from physiobank
        sig_pb, fields_pb = wfdb.rdsamp('100', pb_dir = 'mitdb')
        # This comment line was manually added and is not present in the
        # original physiobank record
        del(fields['comments'][0])

        assert np.array_equal(sig_round, sig_target)
        assert np.array_equal(sig, sig_pb) and fields == fields_pb

    def test_2b(self):
        """
        Format 212, selected duration, selected channel, digital.

        Target file created with:
            rdsamp -r sample-data/100 -f 0.002 -t 30 -s 1 | cut -f 2- > record-2b
        """
        record = wfdb.rdrecord('sample-data/100', sampfrom=1,
                               sampto=10800, channels=[1], physical=False)
        sig = record.d_signal
        sig_target = np.genfromtxt('tests/target-output/record-2b')
        sig_target = sig_target.reshape(len(sig_target), 1)

        # Compare data streaming from physiobank
        record_pb = wfdb.rdrecord('100', sampfrom=1, sampto=10800,
                                  channels=[1], physical=False, pb_dir='mitdb')
        # This comment line was manually added and is not present in the
        # original physiobank record
        del(record.comments[0])

        # Option of selecting channels by name
        record_named = wfdb.rdrecord('sample-data/100', sampfrom=1,
                                     sampto=10800, channel_names=['V5'],
                                     physical=False)
        del(record_named.comments[0])

        # Test file writing
        record.wrsamp()
        record_write = wfdb.rdrecord('100', physical=False)

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pb)
        assert record.__eq__(record_named)
        assert record.__eq__(record_write)

    def test_2c(self):
        """
        Format 212, entire signal, physical, odd sampled record.

        Target file created with:
            rdsamp -r sample-data/100_3chan -P | cut -f 2- > record-2c
        """
        record = wfdb.rdrecord('sample-data/100_3chan')
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-2c')

        # Test file writing
        record.d_signal = record.adc()
        record.wrsamp()
        record_write = wfdb.rdrecord('100_3chan')
        record.d_signal = None

        assert np.array_equal(sig_round, sig_target)
        assert record.__eq__(record_write)

    def test_2d(self):
        """
        Format 310, selected duration, digital
        Target file created with:
            rdsamp -r sample-data/3000003_0003 -f 0 -t 8.21 | cut -f 2- | wrsamp -o 310derive -O 310
            rdsamp -r 310derive -f 0.007 | cut -f 2- > record-2d
        """
        record = wfdb.rdrecord('sample-data/310derive', sampfrom=2,
                               physical=False)
        sig = record.d_signal
        sig_target = np.genfromtxt('tests/target-output/record-2d')
        assert np.array_equal(sig, sig_target)

    def test_2e(self):
        """
        Format 311, selected duration, physical.

        Target file created with:
            rdsamp -r sample-data/3000003_0003 -f 0 -t 8.21 -s 1 | cut -f 2- | wrsamp -o 311derive -O 311
            rdsamp -r 311derive -f 0.005 -t 3.91 -P | cut -f 2- > record-2e
        """
        sig, fields = wfdb.rdsamp('sample-data/311derive', sampfrom=1,
                                  sampto=978)
        sig = np.round(sig, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-2e')
        sig_target = sig_target.reshape([977, 1])
        assert np.array_equal(sig, sig_target)


    # --------------------- 3. Multi-dat records --------------------- #

    def test_3a(self):
        """
        Multi-dat, entire signal, digital
        Target file created with:
            rdsamp -r sample-data/s0010_re | cut -f 2- > record-3a
        """
        record= wfdb.rdrecord('sample-data/s0010_re', physical=False)
        sig = record.d_signal
        sig_target = np.genfromtxt('tests/target-output/record-3a')

        # Compare data streaming from physiobank
        record_pb= wfdb.rdrecord('s0010_re', physical=False,
                                 pb_dir='ptbdb/patient001')

        # Test file writing
        record.wrsamp()
        record_write = wfdb.rdrecord('s0010_re', physical=False)

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pb)
        assert record.__eq__(record_write)

    def test_3b(self):
        """
        Multi-dat, selected duration, selected channels, physical.

        Target file created with:
            rdsamp -r sample-data/s0010_re -f 5 -t 38 -P -s 13 0 4 8 3 | cut -f 2- > record-3b
        """
        sig, fields = wfdb.rdsamp('sample-data/s0010_re', sampfrom=5000,
                                  sampto=38000, channels=[13, 0, 4, 8, 3])
        sig_round = np.round(sig, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-3b')

        # Compare data streaming from physiobank
        sig_pb, fields_pb = wfdb.rdsamp('s0010_re', sampfrom=5000,
                                        pb_dir='ptbdb/patient001',
                                        sampto=38000,
                                        channels=[13, 0, 4, 8, 3])

        assert np.array_equal(sig_round, sig_target)
        assert np.array_equal(sig, sig_pb) and fields == fields_pb


    # -------------- 4. Skew and multiple samples/frame -------------- #

    def test_4a(self):
        """
        Format 16, multi-samples per frame, skew, digital.

        Target file created with:
            rdsamp -r sample-data/test01_00s_skewframe | cut -f 2- > record-4a
        """
        record = wfdb.rdrecord('sample-data/test01_00s_skewframe',
                               physical=False)
        sig = record.d_signal
        # The WFDB library rdsamp does not return the final N samples for all
        # channels due to the skew. The WFDB python rdsamp does return the final
        # N samples, filling in NANs for end of skewed channels only.
        sig = sig[:-3, :]

        sig_target = np.genfromtxt('tests/target-output/record-4a')

        # Test file writing. Multiple samples per frame and skew.
        # Have to read all the samples in the record, ignoring skew
        record_no_skew = wfdb.rdrecord('sample-data/test01_00s_skewframe',
                                       physical=False,
                                       smooth_frames=False, ignore_skew=True)
        record_no_skew.wrsamp(expanded=True)
        # Read the written record
        record_write = wfdb.rdrecord('test01_00s_skewframe', physical=False)

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_write)

    def test_4b(self):
        """
        Format 12, multi-samples per frame, skew, entire signal, digital.

        Target file created with:
            rdsamp -r sample-data/03700181 | cut -f 2- > record-4b
        """
        record = wfdb.rdrecord('sample-data/03700181', physical=False)
        sig = record.d_signal
        # The WFDB library rdsamp does not return the final N samples for all
        # channels due to the skew.
        sig = sig[:-4, :]
        # The WFDB python rdsamp does return the final N samples, filling in
        # NANs for end of skewed channels only.
        sig_target = np.genfromtxt('tests/target-output/record-4b')

        # Compare data streaming from physiobank
        record_pb = wfdb.rdrecord('03700181', physical=False,
                                  pb_dir='mimicdb/037')

        # Test file writing. Multiple samples per frame and skew.
        # Have to read all the samples in the record, ignoring skew
        record_no_skew = wfdb.rdrecord('sample-data/03700181', physical=False,
                                       smooth_frames=False, ignore_skew=True)
        record_no_skew.wrsamp(expanded=True)
        # Read the written record
        record_write = wfdb.rdrecord('03700181', physical=False)

        assert np.array_equal(sig, sig_target)
        assert record.__eq__(record_pb)
        assert record.__eq__(record_write)

    def test_4c(self):
        """
        Format 12, multi-samples per frame, skew, selected suration,
        selected channels, physical.

        Target file created with:
            rdsamp -r sample-data/03700181 -f 8 -t 128 -s 0 2 -P | cut -f 2- > record-4c
        """
        sig, fields = wfdb.rdsamp('sample-data/03700181', channels=[0, 2],
                                  sampfrom=1000, sampto=16000)
        sig_round = np.round(sig, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-4c')

        # Compare data streaming from physiobank
        sig_pb, fields_pb = wfdb.rdsamp('03700181', pb_dir='mimicdb/037',
                                        channels=[0, 2], sampfrom=1000,
                                        sampto=16000)

        # Test file writing. Multiple samples per frame and skew.
        # Have to read all the samples in the record, ignoring skew
        record_no_skew = wfdb.rdrecord('sample-data/03700181', physical=False,
                                       smooth_frames=False, ignore_skew=True)
        record_no_skew.wrsamp(expanded=True)
        # Read the written record
        writesig, writefields = wfdb.rdsamp('03700181', channels=[0, 2],
                                            sampfrom=1000, sampto=16000)

        assert np.array_equal(sig_round, sig_target)
        assert np.array_equal(sig, sig_pb) and fields == fields_pb
        assert np.array_equal(sig, writesig) and fields == writefields

    def test_4d(self):
        """
        Format 16, multi-samples per frame, skew, read expanded signals

        Target file created with:
            rdsamp -r sample-data/test01_00s_skewframe -P -H | cut -f 2- > record-4d
        """
        record = wfdb.rdrecord('sample-data/test01_00s_skewframe',
                               smooth_frames=False)

        # Upsample the channels with lower samples/frame
        expandsig = np.zeros((7994, 3))
        expandsig[:,0] = np.repeat(record.e_p_signal[0][:-3],2)
        expandsig[:,1] = record.e_p_signal[1][:-6]
        expandsig[:,2] = np.repeat(record.e_p_signal[2][:-3],2)

        sig_round = np.round(expandsig, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-4d')

        assert np.array_equal(sig_round, sig_target)


    @classmethod
    def tearDownClass(cls):
        "Clean up written files"
        writefiles = ['03700181.dat','03700181.hea','100.atr','100.dat',
                      '100.hea','1003.atr','100_3chan.dat','100_3chan.hea',
                      '12726.anI','a103l.hea','a103l.mat','s0010_re.dat',
                      's0010_re.hea','s0010_re.xyz','test01_00s.dat',
                      'test01_00s.hea','test01_00s_skewframe.hea']

        for file in writefiles:
            if os.path.isfile(file):
                os.remove(file)


class TestMultiRecord():
    """
    Test read and write of multi segment wfdb records, including
    Physionet streaming.

    Target files created using the original WFDB Software Package
    version 10.5.24

    """

    def test_multi_fixed_a(self):
        """
        Multi-segment, fixed layout, read entire signal.

        Target file created with:
            rdsamp -r sample-data/multi-segment/fixed1/v102s -P | cut -f 2- > record-multi-fixed-a
        """
        record = wfdb.rdrecord('sample-data/multi-segment/fixed1/v102s')
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-multi-fixed-a')

        np.testing.assert_equal(sig_round, sig_target)

    def test_multi_fixed_b(self):
        """
        Multi-segment, fixed layout, selected duration, samples read
        from one segment.

        Target file created with:
            rdsamp -r sample-data/multi-segment/fixed1/v102s -t s75000 -P | cut -f 2- > record-multi-fixed-b
        """
        record = wfdb.rdrecord('sample-data/multi-segment/fixed1/v102s',
                               sampto=75000)
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-multi-fixed-b')

        np.testing.assert_equal(sig_round, sig_target)

    def test_multi_fixed_c(self):
        """
        Multi-segment, fixed layout, selected duration and channels,
        samples read from multiple segments

        Target file created with:
            rdsamp -r sample-data/multi-segment/fixed1/v102s -f s70000 -t s80000 -s 1 0 3 -P | cut -f 2- > record-multi-fixed-c
        """
        record = wfdb.rdrecord('sample-data/multi-segment/fixed1/v102s',
                               sampfrom=70000, sampto=80000, channels=[1, 0, 3])
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-multi-fixed-c')

        # Option of selecting channels by name
        record_named = wfdb.rdrecord('sample-data/multi-segment/fixed1/v102s',
                                     sampfrom=70000, sampto=80000,
                                     channel_names=['V', 'II', 'RESP'])

        np.testing.assert_equal(sig_round, sig_target)
        assert record.__eq__(record_named)

    def test_multi_variable_a(self):
        """
        Multi-segment, variable layout, selected duration, samples read
        from one segment only.

        Target file created with:
            rdsamp -r sample-data/multi-segment/s00001/s00001-2896-10-10-00-31 -f s14428365 -t s14428375 -P | cut -f 2- > record-multi-variable-a
        """
        record = wfdb.rdrecord('sample-data/multi-segment/s00001/s00001-2896-10-10-00-31',
                               sampfrom=14428365, sampto=14428375)
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-multi-variable-a')

        np.testing.assert_equal(sig_round, sig_target)

    def test_multi_variable_b(self):
        """
        Multi-segment, variable layout, selected duration, samples read
        from several segments.

        Target file created with:
        rdsamp -r sample-data/multi-segment/s00001/s00001-2896-10-10-00-31 -f s14428364 -t s14428375 -P | cut -f 2- > record-multi-variable-b
        """
        record = wfdb.rdrecord('sample-data/multi-segment/s00001/s00001-2896-10-10-00-31',
                               sampfrom=14428364, sampto=14428375)
        sig_round = np.round(record.p_signal, decimals=8)
        sig_target = np.genfromtxt('tests/target-output/record-multi-variable-b')

        np.testing.assert_equal(sig_round, sig_target)

    def test_multi_variable_c(self):
        """
        Multi-segment, variable layout, entire signal, physical

        The reference signal creation cannot be made with rdsamp
        directly because the wfdb c package (10.5.24) applies the single
        adcgain and baseline values from the layout specification
        header, which is undesired in multi-segment signals with
        different adcgain/baseline values across segments.

        Target file created with:
        ```
        for i in {01..18}
        do
            rdsamp -r sample-data/multi-segment/s25047/3234460_00$i -P | cut -f 2- >> record-multi-variable-c
        done
        ```

        Entire signal has 543240 samples.
        - 25740 length empty segment.
        - First 16 segments have same 2 channels, length 420000
        - Last 2 segments have same 3 channels, length 97500

        """
        record = wfdb.rdrecord('sample-data/multi-segment/s25047/s25047-2704-05-04-10-44')
        sig_round = np.round(record.p_signal, decimals=8)

        sig_target_a = np.full((25740,3), np.nan)
        sig_target_b = np.concatenate(
            (np.genfromtxt('tests/target-output/record-multi-variable-c', skip_footer=97500),
             np.full((420000, 1), np.nan)), axis=1)
        sig_target_c = np.genfromtxt('tests/target-output/record-multi-variable-c',
                                     skip_header=420000)
        sig_target = np.concatenate((sig_target_a, sig_target_b, sig_target_c))

        np.testing.assert_equal(sig_round, sig_target)

    def test_multi_variable_d(self):
        """
        Multi-segment, variable layout, selected duration, selected
        channels, digital. There are two channels: PLETH, and II. Their
        fmt, adc_gain, and baseline do not change between the segments.

        Target file created with:
            rdsamp -r sample-data/multi-segment/p000878/p000878-2137-10-26-16-57 -f s3550 -t s7500 -s 0 1 | cut -f 2- | perl -p -e 's/-32768/  -128/g;' > record-multi-variable-d

        """
        record = wfdb.rdrecord('sample-data/multi-segment/p000878/p000878-2137-10-26-16-57',
                                sampfrom=3550, sampto=7500, channels=[0, 1],
                                physical=False)
        sig = record.d_signal

        # Compare data streaming from physiobank
        record_pb = wfdb.rdrecord('p000878-2137-10-26-16-57',
                                  pb_dir='mimic3wdb/matched/p00/p000878/',
                                  sampfrom=3550, sampto=7500, channels=[0, 1],
                                  physical=False)
        sig_target = np.genfromtxt('tests/target-output/record-multi-variable-d')

        # Option of selecting channels by name
        record_named = wfdb.rdrecord('sample-data/multi-segment/p000878/p000878-2137-10-26-16-57',
                                     sampfrom=3550, sampto=7500, physical=False,
                                     channel_names=['PLETH', 'II'])


        np.testing.assert_equal(sig, sig_target)
        assert record.__eq__(record_pb)
        assert record.__eq__(record_named)


class TestSignal():
    """
    For lower level signal tests

    """
    def test_infer_sig_len(self):
        """
        Infer the signal length of a record without the sig_len header
        Read two headers. The records should be the same.
        """

        record = wfdb.rdrecord('sample-data/100')
        record_2 = wfdb.rdrecord('sample-data/100-no-len')
        record_2.record_name = '100'

        assert record_2.__eq__(record)


class TestDownload():
    # Test that we can download records with no "dat" file
    # Regression test for https://github.com/MIT-LCP/wfdb-python/issues/118
    def test_dl_database_no_dat_file(self):
        wfdb.dl_database('afdb', './download-tests/', ['00735'])

    # Test that we can download records that *do* have a "dat" file.
    def test_dl_database_with_dat_file(self):
        wfdb.dl_database('afdb', './download-tests/', ['04015'])

    # Cleanup written files
    @classmethod
    def tearDownClass(self):
        if os.path.isdir('./download-tests/'):
            shutil.rmtree('./download-tests/')
