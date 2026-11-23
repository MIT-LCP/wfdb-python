**Fix get_dbs, which PhysioNet no longer supported**
  ``get_dbs`` failed with a 404 error, because PhysioNet has removed the
  ``/rest/database-list/`` page that it read. It now lists databases through
  PhysioNet's API, using the ``physionet`` package, and returns the slug and
  title of the latest version of each database, as before.
