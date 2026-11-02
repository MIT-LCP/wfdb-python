**Allow empty annotations in compare_annotations**
  ``compare_annotations`` (``Comparitor.compare``) no longer fails when the reference or test annotations are empty, for example when a detector finds no beats. The sensitivity or positive predictivity that cannot be computed is NaN.
