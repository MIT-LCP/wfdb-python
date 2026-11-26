**Skip folders and files when checking a dataset**
  ``check_dataset`` has an ``ignore`` argument, a function that returns True
  for paths to skip. It chooses which headers and archives to check, and the
  contents of skipped folders are not read. Signal files and segments that a
  checked record refers to are still checked.
