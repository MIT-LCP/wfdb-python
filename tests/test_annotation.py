import re

import numpy as np

import wfdb

class test_annotation():
    """
    Testing read and write of wfdb annotations, including Physionet
    streaming.

    Target files created using the original WFDB Software Package
    version 10.5.24
    """

    def test_1(self):
        """
        Target file created with:
            rdann -r sample-data/100 -a atr > ann-1
        """
        annotation = wfdb.rdann('sample-data/100', 'atr')


        # This is not the fault of the script. The annotation file specifies a
        # length 3
        annotation.aux_note[0] = '(N'
        # aux_note field with a null written after '(N' which the script correctly picks up. I am just
        # getting rid of the null in this unit test to compare with the regexp output below which has
        # no null to detect in the output text file of rdann.

        # Target data from WFDB software package
        lines = tuple(open('tests/target-output/ann-1', 'r'))
        nannot = len(lines)

        target_time = [None] * nannot
        target_sample = np.empty(nannot, dtype='object')
        target_symbol = [None] * nannot
        target_subtype = np.empty(nannot, dtype='object')
        target_chan = np.empty(nannot, dtype='object')
        target_num = np.empty(nannot, dtype='object')
        target_aux_note = [None] * nannot

        RXannot = re.compile(
            '[ \t]*(?P<time>[\[\]\w\.:]+) +(?P<sample>\d+) +(?P<symbol>.) +(?P<subtype>\d+) +(?P<chan>\d+) +(?P<num>\d+)\t?(?P<aux_note>.*)')

        for i in range(0, nannot):
            target_time[i], target_sample[i], target_symbol[i], target_subtype[i], target_chan[
                i], target_num[i], target_aux_note[i] = RXannot.findall(lines[i])[0]

        # Convert objects into integers
        target_sample = target_sample.astype('int')
        target_num = target_num.astype('int')
        target_subtype = target_subtype.astype('int')
        target_chan = target_chan.astype('int')

        # Compare
        comp = [np.array_equal(annotation.sample, target_sample),
                np.array_equal(annotation.symbol, target_symbol),
                np.array_equal(annotation.subtype, target_subtype),
                np.array_equal(annotation.chan, target_chan),
                np.array_equal(annotation.num, target_num),
                annotation.aux_note == target_aux_note]

        # Test file streaming
        pbannotation = wfdb.rdann('100', 'atr', pb_dir='mitdb', return_label_elements=['label_store', 'symbol'])
        pbannotation.aux_note[0] = '(N'
        pbannotation.create_label_map()

        # Test file writing
        annotation.wrann(write_fs=True)
        writeannotation = wfdb.rdann('100', 'atr', return_label_elements=['label_store', 'symbol'])
        writeannotation.create_label_map()

        assert (comp == [True] * 6)
        assert annotation.__eq__(pbannotation)
        assert annotation.__eq__(writeannotation)

    def test_2(self):
        """
        Annotation file with many aux_note strings.

        Target file created with:
            rdann -r sample-data/100 -a atr > ann-2
        """
        annotation = wfdb.rdann('sample-data/12726', 'anI')

        # Target data from WFDB software package
        lines = tuple(open('tests/target-output/ann-2', 'r'))
        nannot = len(lines)

        target_time = [None] * nannot
        target_sample = np.empty(nannot, dtype='object')
        target_symbol = [None] * nannot
        target_subtype = np.empty(nannot, dtype='object')
        target_chan = np.empty(nannot, dtype='object')
        target_num = np.empty(nannot, dtype='object')
        target_aux_note = [None] * nannot

        RXannot = re.compile(
            '[ \t]*(?P<time>[\[\]\w\.:]+) +(?P<sample>\d+) +(?P<symbol>.) +(?P<subtype>\d+) +(?P<chan>\d+) +(?P<num>\d+)\t?(?P<aux_note>.*)')

        for i in range(0, nannot):
            target_time[i], target_sample[i], target_symbol[i], target_subtype[i], target_chan[
                i], target_num[i], target_aux_note[i] = RXannot.findall(lines[i])[0]

        # Convert objects into integers
        target_sample = target_sample.astype('int')
        target_num = target_num.astype('int')
        target_subtype = target_subtype.astype('int')
        target_chan = target_chan.astype('int')

        # Compare
        comp = [np.array_equal(annotation.sample, target_sample),
                np.array_equal(annotation.symbol, target_symbol),
                np.array_equal(annotation.subtype, target_subtype),
                np.array_equal(annotation.chan, target_chan),
                np.array_equal(annotation.num, target_num),
                annotation.aux_note == target_aux_note]
        # Test file streaming
        pbannotation = wfdb.rdann('12726', 'anI', pb_dir='prcp', return_label_elements=['label_store', 'symbol'])
        pbannotation.create_label_map()

        # Test file writing
        annotation.wrann(write_fs=True)
        writeannotation = wfdb.rdann('12726', 'anI', return_label_elements=['label_store', 'symbol'])
        writeannotation.create_label_map()

        assert (comp == [True] * 6)
        assert annotation.__eq__(pbannotation)
        assert annotation.__eq__(writeannotation)

    def test_3(self):
        """
        Annotation file with custom annotation types

        Target file created with:
            rdann -r sample-data/1003 -a atr > ann-3
        """
        annotation = wfdb.rdann('sample-data/1003', 'atr')

        # Target data from WFDB software package
        lines = tuple(open('tests/target-output/ann-3', 'r'))
        nannot = len(lines)

        target_time = [None] * nannot
        target_sample = np.empty(nannot, dtype='object')
        target_symbol = [None] * nannot
        target_subtype = np.empty(nannot, dtype='object')
        target_chan = np.empty(nannot, dtype='object')
        target_num = np.empty(nannot, dtype='object')
        target_aux_note = [None] * nannot

        RXannot = re.compile(
            '[ \t]*(?P<time>[\[\]\w\.:]+) +(?P<sample>\d+) +(?P<symbol>.) +(?P<subtype>\d+) +(?P<chan>\d+) +(?P<num>\d+)\t?(?P<aux_note>.*)')

        for i in range(0, nannot):
            target_time[i], target_sample[i], target_symbol[i], target_subtype[i], target_chan[
                i], target_num[i], target_aux_note[i] = RXannot.findall(lines[i])[0]

        # Convert objects into integers
        target_sample = target_sample.astype('int')
        target_num = target_num.astype('int')
        target_subtype = target_subtype.astype('int')
        target_chan = target_chan.astype('int')

        # Compare
        comp = [np.array_equal(annotation.sample, target_sample),
                np.array_equal(annotation.symbol, target_symbol),
                np.array_equal(annotation.subtype, target_subtype),
                np.array_equal(annotation.chan, target_chan),
                np.array_equal(annotation.num, target_num),
                annotation.aux_note == target_aux_note]

        # Test file streaming
        pbannotation = wfdb.rdann('1003', 'atr', pb_dir='challenge/2014/set-p2', return_label_elements=['label_store', 'symbol'])
        pbannotation.create_label_map()

        # Test file writing
        annotation.wrann(write_fs=True)
        writeannotation = wfdb.rdann('1003', 'atr', return_label_elements=['label_store', 'symbol'])
        writeannotation.create_label_map()

        assert (comp == [True] * 6)
        assert annotation.__eq__(pbannotation)
        assert annotation.__eq__(writeannotation)
