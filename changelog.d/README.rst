Changelog entries
=================

Each user-visible change gets its own file in this directory, instead of an
entry in ``docs/changes.rst``. Separate files avoid merge conflicts between
pull requests. At release time, ``towncrier build`` collects them into
``docs/changes.rst`` and deletes them.

Name the file after the change, with an ``.rst`` extension and no other dots
(for example ``rr2ann-fixes.rst``). Write the entry as a bold heading followed
by an indented paragraph, as in ``docs/changes.rst``::

  **Fix rr2ann with times**
    ``rr2ann`` with ``as_time=True`` now rounds times to the nearest sample.

To preview the next release's changelog, run::

  uv run towncrier build --draft
