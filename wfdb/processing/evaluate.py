import numpy as np

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

        self.fp = 0
        self.tp = 0

        self.n_missed = 0
        self.n_detected = 0

        # How many there are
        self.n_ref = len(ref_sample)
        self.n_comp = len(test_sample)


        # # Just derive these 4 at the end?
        # # Index info about the reference samples
        # self.detected_inds = []
        # self.missed_inds = []
        # # About the testing samples
        # self.correct_test_inds = []
        # self.wrong_test_inds = []


        # The matching test sample numbers. -1 for indices with no match
        self.matching_sample_nums = -1 * np.ones(n_ref)

        # TODO: rdann return annotations.where

    def compare(self):



        test_samp_num = 0
        ref_samp_num = 0
        
        while ref_samp_num < n_ref:

            closest_samp_num, smallest_samp_diff = (
                self.get_closest_samp_num(ref_samp_num, test_samp_num))
            closest_samp_num_next, smallest_samp_diff_next = (
                self.get_closest_samp_num(ref_samp_num + 1, test_samp_num))

            # Found a contested test sample number. Decide which reference
            # sample it belongs to.
            if closest_samp_num == closest_samp_num_next:
                pass
            # No clash. Assign the reference-test pair
            else:
                self.matching_sample_nums[ref_samp_num] = closest_samp_num

                ref_samp_num += 1
                test_samp_num = closest_samp_num + 1


        self.calc_stats()

            
        def get_closest_samp_num(self, ref_samp_num, start_test_samp_num):
            """
            Return the closest testing sample number for the given reference
            sample number. Begin the search from start_test_samp_num.
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







def compare_annotations(ind_ref, ind_comp):
    """

    """


    detected_inds
    missed_inds


    tp
    tn
    fp
    fn

    tpr
    tnr
    fpr
    fnr

    return evaluation


def plot_comparitor(comparitor, sig=None):
    """
    Plot two sets of annotations

    Parameters
    ----------
    comparitor: Comparitor object
        The comparitor object that has been run
    sig : numpy array, optional
        The underlying signal of the two sets of annotations to plot.
    """
    return
