import numpy as np
import matplotlib.pyplot as plt


class Comparitor(object):
    """
    The class to implement and hold comparisons between two sets of
    annotations.

    See methods `print_summary` and `plot`.

    Examples
    --------
    >>> import wfdb
    >>> from wfdb import processing

    >>> sig, fields = wfdb.rdsamp('sample-data/100', channels=[0])
    >>> ann_ref = wfdb.rdann('sample-data/100','atr')
    >>> xqrs = processing.XQRS(sig=sig[:,0], fs=fields['fs'])
    >>> xqrs.detect()

    >>> comparitor = processing.Comparitor(ann_ref.sample[1:],
                                           xqrs.qrs_inds,
                                           int(0.1 * fields['fs']),
                                           sig[:,0])
    >>> comparitor.compare()
    >>> comparitor.print_summary()
    >>> comparitor.plot()

    """
    def __init__(self, ref_sample, test_sample, window_width, signal=None):
        """
        Parameters
        ----------
        ref_sample : numpy array
            An array of the reference sample locations
        test_sample : numpy array
            An array of the comparison sample locations
        window_width : int
            The width of the window
        signal : 1d numpy array, optional
            The signal array the annotation samples are labelling. Only used
            for plotting.
        """
        if min(np.diff(ref_sample)) < 0 or min(np.diff(test_sample)) < 0:
            raise ValueError(('The sample locations must be monotonically'
                              + ' increasing'))

        self.ref_sample = ref_sample
        self.test_sample = test_sample
        self.n_ref = len(ref_sample)
        self.n_test = len(test_sample)
        self.window_width = window_width

        # The matching test sample number for each reference annotation.
        # -1 for indices with no match
        self.matching_sample_nums = -1 * np.ones(self.n_ref, dtype='int')

        self.signal = signal
        # TODO: rdann return annotations.where

    def _calc_stats(self):
        """
        Calculate performance statistics after the two sets of annotations
        are compared.

        Example:
        -------------------
         ref=500  test=480
        {  30 { 470 } 10  }
        -------------------

        tp = 470
        fp = 10
        fn = 30

        specificity = 470 / 500
        positive_predictivity = 470 / 480
        false_positive_rate = 10 / 480

        """
        # Reference annotation indices that were detected
        self.matched_ref_inds = np.where(self.matching_sample_nums != -1)[0]
        # Reference annotation indices that were missed
        self.unmatched_ref_inds = np.where(self.matching_sample_nums == -1)[0]
        # Test annotation indices that were matched to a reference annotation
        self.matched_test_inds = self.matching_sample_nums[
            self.matching_sample_nums != -1]
        # Test annotation indices that were unmatched to a reference annotation
        self.unmatched_test_inds = np.setdiff1d(np.array(range(self.n_test)),
            self.matched_test_inds, assume_unique=True)

        # Sample numbers that were matched and unmatched
        self.matched_ref_sample = self.ref_sample[self.matched_ref_inds]
        self.unmatched_ref_sample = self.ref_sample[self.unmatched_ref_inds]
        self.matched_test_sample = self.test_sample[self.matched_test_inds]
        self.unmatched_test_sample = self.test_sample[self.unmatched_test_inds]

        # True positives = matched reference samples
        self.tp = len(self.matched_ref_inds)
        # False positives = extra test samples not matched
        self.fp = self.n_test - self.tp
        # False negatives = undetected reference samples
        self.fn = self.n_ref - self.tp
        # No tn attribute

        self.specificity = float(self.tp) / self.n_ref
        self.positive_predictivity = float(self.tp) / self.n_test
        self.false_positive_rate = float(self.fp) / self.n_test


    def compare(self):
        """
        Main comparison function
        """
        test_samp_num = 0
        ref_samp_num = 0

        # Iterate through the reference sample numbers
        while ref_samp_num < self.n_ref and test_samp_num < self.n_test:

            # Get the closest testing sample number for this reference sample
            closest_samp_num, smallest_samp_diff = (
                self._get_closest_samp_num(ref_samp_num, test_samp_num))
            # Get the closest testing sample number for the next reference
            # sample. This doesn't need to be called for the last index.
            if ref_samp_num < self.n_ref - 1:
                closest_samp_num_next, smallest_samp_diff_next = (
                    self._get_closest_samp_num(ref_samp_num + 1, test_samp_num))
            else:
                # Set non-matching value if there is no next reference sample
                # to compete for the test sample
                closest_samp_num_next = -1

            # Found a contested test sample number. Decide which reference
            # sample it belongs to.
            if closest_samp_num == closest_samp_num_next:
                # If the sample is closer to the next reference sample,
                # assign it to the next refernece sample.
                if smallest_samp_diff_next < smallest_samp_diff:
                    # Get the next closest sample for this reference sample.
                    # Can this be empty? Need to catch case where nothing left?
                    closest_samp_num, smallest_samp_diff = (
                        self._get_closest_samp_num(ref_samp_num, test_samp_num))

            # If no clash, it is straightforward.

            # Assign the reference-test pair if close enough
            if smallest_samp_diff < self.window_width:
                self.matching_sample_nums[ref_samp_num] = closest_samp_num

            ref_samp_num += 1
            test_samp_num = closest_samp_num + 1

        self._calc_stats()


    def _get_closest_samp_num(self, ref_samp_num, start_test_samp_num):
        """
        Return the closest testing sample number for the given reference
        sample number. Limit the search from start_test_samp_num.
        """

        if start_test_samp_num >= self.n_test:
            raise ValueError('Invalid starting test sample number.')

        ref_samp = self.ref_sample[ref_samp_num]
        test_samp = self.test_sample[start_test_samp_num]
        samp_diff = ref_samp - test_samp

        # Initialize running parameters
        closest_samp_num = start_test_samp_num
        smallest_samp_diff = abs(samp_diff)

        # Iterate through the testing samples
        for test_samp_num in range(start_test_samp_num, self.n_test):
            test_samp = self.test_sample[test_samp_num]
            samp_diff = ref_samp - test_samp
            abs_samp_diff = abs(samp_diff)

            # Found a better match
            if abs(samp_diff) < smallest_samp_diff:
                closest_samp_num = test_samp_num
                smallest_samp_diff = abs_samp_diff

            # Stop iterating when the ref sample is first passed or reached
            if samp_diff <= 0:
                break

        return closest_samp_num, smallest_samp_diff

    def print_summary(self):
        """
        Print summary metrics of the annotation comparisons.
        """
        # True positives = matched reference samples
        self.tp = len(self.matched_ref_inds)
        # False positives = extra test samples not matched
        self.fp = self.n_test - self.tp
        # False negatives = undetected reference samples
        self.fn = self.n_ref - self.tp
        # No tn attribute

        self.specificity = self.tp / self.n_ref
        self.positive_predictivity = self.tp / self.n_test
        self.false_positive_rate = self.fp / self.n_test

        print('%d reference annotations, %d test annotations\n'
            % (self.n_ref, self.n_test))
        print('True Positives (matched samples): %d' % self.tp)
        print('False Positives (unmatched test samples: %d' % self.fp)
        print('False Negatives (unmatched reference samples): %d\n' % self.fn)

        print('Specificity: %.4f (%d/%d)'
            % (self.specificity, self.tp, self.n_ref))
        print('Positive Predictivity: %.4f (%d/%d)'
            % (self.positive_predictivity, self.tp, self.n_test))
        print('False Positive Rate: %.4f (%d/%d)'
            % (self.false_positive_rate, self.fp, self.n_test))


    def plot(self, sig_style='', title=None, figsize=None,
             return_fig=False):
        """
        Plot the comparison of two sets of annotations, possibly
        overlaid on their original signal.

        Parameters
        ----------
        sig_style : str, optional
            The matplotlib style of the signal
        title : str, optional
            The title of the plot
        figsize: tuple, optional
            Tuple pair specifying the width, and height of the figure.
            It is the'figsize' argument passed into matplotlib.pyplot's
            `figure` function.
        return_fig : bool, optional
            Whether the figure is to be returned as an output argument.

        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        legend = ['Signal',
                  'Matched Reference Annotations (%d/%d)' % (self.tp, self.n_ref),
                  'Unmatched Reference Annotations (%d/%d)' % (self.fn, self.n_ref),
                  'Matched Test Annotations (%d/%d)' % (self.tp, self.n_test),
                  'Unmatched Test Annotations (%d/%d)' % (self.fp, self.n_test)
        ]

        # Plot the signal if any
        if self.signal is not None:
            ax.plot(self.signal, sig_style)

            # Plot reference annotations
            ax.plot(self.matched_ref_sample,
                    self.signal[self.matched_ref_sample], 'ko')
            ax.plot(self.unmatched_ref_sample,
                    self.signal[self.unmatched_ref_sample], 'ko',
                    fillstyle='none')
            # Plot test annotations
            ax.plot(self.matched_test_sample,
                    self.signal[self.matched_test_sample], 'g+')
            ax.plot(self.unmatched_test_sample,
                    self.signal[self.unmatched_test_sample], 'rx')

            ax.legend(legend)

        # Just plot annotations
        else:
           # Plot reference annotations
            ax.plot(self.matched_ref_sample, np.ones(self.tp), 'ko')
            ax.plot(self.unmatched_ref_sample, np.ones(self.fn), 'ko',
                fillstyle='none')
            # Plot test annotations
            ax.plot(self.matched_test_sample, 0.5 * np.ones(self.tp), 'g+')
            ax.plot(self.unmatched_test_sample, 0.5 * np.ones(self.fp), 'rx')
            ax.legend(legend[1:])

        if title:
            ax.set_title(title)

        ax.set_xlabel('time/sample')

        fig.show()

        if return_fig:
            return fig, ax


def compare_annotations(ref_sample, test_sample, window_width, signal=None):
    """
    Compare a set of reference annotation locations against a set of
    test annotation locations.

    See the Comparitor class  docstring for more information.

    Parameters
    ----------
    ref_sample : 1d numpy array
        Array of reference sample locations
    test_sample : 1d numpy array
        Array of test sample locations to compare
    window_width : int
        The maximum absolute difference in sample numbers that is
        permitted for matching annotations.
    signal : 1d numpy array, optional
        The original signal of the two annotations. Only used for
        plotting.

    Returns
    -------
    comparitor : Comparitor object
        Object containing parameters about the two sets of annotations

    Examples
    --------
    >>> import wfdb
    >>> from wfdb import processing

    >>> sig, fields = wfdb.rdsamp('sample-data/100', channels=[0])
    >>> ann_ref = wfdb.rdann('sample-data/100','atr')
    >>> xqrs = processing.XQRS(sig=sig[:,0], fs=fields['fs'])
    >>> xqrs.detect()

    >>> comparitor = processing.compare_annotations(ann_ref.sample[1:],
                                                    xqrs.qrs_inds,
                                                    int(0.1 * fields['fs']),
                                                    sig[:,0])
    >>> comparitor.print_summary()
    >>> comparitor.plot()

    """
    comparitor = Comparitor(ref_sample=ref_sample, test_sample=test_sample,
                            window_width=window_width, signal=signal)
    comparitor.compare()

    return comparitor
