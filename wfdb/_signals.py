import numpy as np
import re
import os
import sys
import _headers


# Class with signal method definitions.
# To be inherited by WFDBrecord from records.py.
class Signals_Mixin():

    def wrdats(self):



    # Check the cohesion of the d_signals field with the other fields needed to write the record
    def checksignalcohesion(self, writefields):
