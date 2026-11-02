**Fix get_contained_labels with pandas 3**
  ``Annotation.get_contained_labels`` no longer raises a ``TypeError`` with pandas 3 for annotations without ``label_store`` values, such as those read with ``rdann`` by default.
