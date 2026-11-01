**Add timeouts to requests for remote files**
  Requests for files on PhysioNet (and other web servers) now time out if the server does not respond (10 seconds to connect, and 60 seconds to wait for data), instead of waiting indefinitely. A ``requests.exceptions.Timeout`` (``ConnectTimeout`` or ``ReadTimeout``) is raised in this case.
