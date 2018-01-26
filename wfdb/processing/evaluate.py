import numpy as np
import matplotlib.pyplot as plt


class Comparitor(object):

    def __init__(self, ref_sample, test_sample, window_width):
        """
        Parameters
        ----------
        ref_sample : numpy array
            An array of the reference sample locations
        test_sample : numpy array
            An array of the comparison sample locations
        """
        if min(np.diff(ref_sample)) < 0 or min(np.diff(test_sample)) < 0:
            raise ValueError(("The sample locations must be monotonically"
                              + " increasing"))
        
        self.ref_sample = ref_sample
        self.test_sample = test_sample
        self.n_ref = len(ref_sample)
        self.n_comp = len(test_sample)

        # The matching test sample numbers. -1 for indices with no match
        self.matching_sample_nums = -1 * np.ones(n_ref)

        # TODO: rdann return annotations.where

    def calc_stats(self):
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
        self.detected_ref_inds = np.where(self.matching_sample_nums != -1)
        self.missed_ref_inds = np.where(self.matching_sample_nums == -1)
        self.matched_test_inds = self.matching_sample_nums(
            self.matching_sample_nums != -1)
        self.unmached_test_inds = np.setdiff1d(np.array(range(self.n_test)),
            self.matched_test_inds, assume_unique=True)

        # True positives = matched reference samples
        self.tp = len(detected_ref_inds)
        # False positives = extra test samples not matched
        self.fp = self.n_test - self.tp
        # False negatives = undetected reference samples
        self.fn = self.n_ref - self.tp
        # No tn attribute

        self.specificity = self.tp / self.n_ref
        self.positive_predictivity = self.tp / self.n_test
        self.false_positive_rate = self.fp / self.n_test


    def compare(self):

        test_samp_num = 0
        ref_samp_num = 0
        
        # Why can't this just be a for loop of ref_samp_num?
        while ref_samp_num < n_ref and test_samp_num < n_test:

            closest_samp_num, smallest_samp_diff = (
                self.get_closest_samp_num(ref_samp_num, test_samp_num,
                                          self.n_test))
            # This needs to work for last index
            closest_samp_num_next, smallest_samp_diff_next = (
                self.get_closest_samp_num(ref_samp_num + 1, test_samp_num,
                                          self.n_test))

            # Found a contested test sample number. Decide which reference
            # sample it belongs to.
            if closest_samp_num == closest_samp_num_next:
                # If the sample is closer to the next reference sample, get
                # the next closest sample for this reference sample.
                if smallest_samp_diff_next < smallest_samp_diff:
                    # Get the next closest sample.
                    # Can this be empty? Need to catch case where nothing left?
                    closest_samp_num, smallest_samp_diff = (
                        self.get_closest_samp_num(ref_samp_num, test_samp_num,
                                                  closest_samp_num))


                self.matching_sample_nums[ref_samp_num] = closest_samp_num

            # If no clash, it is straightforward.
            
            # Assign the reference-test pair if close enough
            if smallest_sample_diff < self.window_width:
                self.matching_sample_nums[ref_samp_num] = closest_samp_num

            ref_samp_num += 1
            test_samp_num = closest_samp_num + 1

        self.calc_stats()

            
        def get_closest_samp_num(self, ref_samp_num, start_test_samp_num,
                                 stop_test_samp_num):
            """
            Return the closest testing sample number for the given reference
            sample number. Limit the search between start_test_samp_num and
            stop_test_samp_num.
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
            for test_samp_num in range(start_test_samp_num, stop_test_samp_num):
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


def compare_annotations(ref_sample, test_sample, window_width):
    """
    
    Parameters
    ----------

    Returns
    -------
    comparitor : Comparitor object
        Object containing parameters about the two sets of annotations

    """
    comparitor = Comparitor(ref_sample, test_sample, window_width)
    comparitor.compare()

    return comparitor

# def plot_record(record=None, title=None, annotation=None, time_units='samples',
#             sig_style='', ann_style='r*', plot_ann_sym=False, figsize=None,
#             return_fig=False, ecg_grids=[]): 
def plot_comparitor(comparitor, sig=None, sig_style='', title=None, figsize=None, return_fig=False):
    """
    Plot two sets of annotations

    Parameters
    ----------
    comparitor: Comparitor object
        The comparitor object that has been run
    sig : numpy array, optional
        The underlying signal of the two sets of annotations to plot.
    """
    
    
    fig=plt.figure(figsize=figsize)



    if return_fig:
        return fig