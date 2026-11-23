**Read PhysioNet records whose names include folders**
  ``rdrecord``, ``rdsamp``, ``rdheader`` and ``rdann`` with ``pn_dir`` now
  read a record name that starts with folders, such as
  ``rdrecord("files/p1000/p10000032/s40689238/40689238",
  pn_dir="mimic-iv-ecg/1.0")``, from those folders of the database. They used
  to drop the folders and look for the record at the top of the database,
  which failed with a "file not found" error.
