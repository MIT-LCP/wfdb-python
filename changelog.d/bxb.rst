**Beat-by-beat comparison of annotations, as WFDB bxb does (development feature)**
  ``processing.bxb`` and ``processing.bxb_record`` compare a test annotation
  file with a reference annotation file beat by beat, as the ``bxb`` program
  of the WFDB Software Package does for the ANSI/AAMI EC57 and EC38
  standards. The result holds the comparison matrix and the QRS, VEB and SVEB
  sensitivity and positive predictivity, RMS RR interval error and shutdown
  statistics, and gives the standard and line-format reports of ``bxb``.
  ``processing.sumstats`` combines the results of several records, as WFDB
  ``sumstats`` does. These functions were written from the standards and the
  documentation of ``bxb``, and by comparing results with it, without
  reference to its source code. They are a development feature: their
  interface may change, and their results should be checked against ``bxb``
  before they are used for reporting.
