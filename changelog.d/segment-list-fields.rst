**Validate multi-segment header fields as lists**
  ``seg_name`` and ``seg_len`` must be lists with correctly typed elements.
  Invalid values now raise a clear ``TypeError`` before writing a header.
