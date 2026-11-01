**Fix test annotations matched twice in compare_annotations**
  ``compare_annotations`` (``Comparitor.compare``) could match one test annotation to two reference annotations, which gave a negative number of false positives and a positive predictivity above 1. Each test annotation is now matched at most once. Results are unchanged when no annotation was matched twice.
