import io
import logging
import os
import platform
import re
import threading
import urllib.parse
import urllib.request

from wfdb.version import __version__


# Value for 'buffering' indicating that the entire file should be
# buffered at once.
BUFFER_WHOLE_FILE = -2

# Default buffer size for remote files.
DEFAULT_BUFFER_SIZE = 32768

# Logger for this module.
_LOGGER = logging.getLogger(__name__)

# Pattern that matches the value of the Content-Range response header.
_CONTENT_RANGE_PATTERN = re.compile(
    r"bytes (?:(\d+)-(\d+)|\*)/(?:(\d+)|\*)", re.ASCII | re.IGNORECASE
)

# Global session object.
_SESSION = None
_SESSION_PID = None
_SESSION_LOCK = threading.Lock()


def _get_session():
    """
    Obtain a session object suitable for requesting remote files.

    Parameters
    ----------
    N/A

    Returns
    -------
    session : requests.Session
        A session object.

    """
    import requests
    import requests.adapters

    global _SESSION
    global _SESSION_PID

    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = requests.Session()
            _SESSION.headers["User-Agent"] = " ".join(
                [
                    "%s/%s" % ("wfdb-python", __version__),
                    "%s/%s" % ("python-requests", requests.__version__),
                    "%s/%s"
                    % (
                        platform.python_implementation(),
                        platform.python_version(),
                    ),
                ]
            )
            for protocol in ("http", "https"):
                adapter = requests.adapters.HTTPAdapter(
                    pool_maxsize=2,
                    pool_block=True,
                )
                _SESSION.mount("%s://" % protocol, adapter)

        # Ensure we don't reuse sockets after forking
        if _SESSION_PID != os.getpid():
            _SESSION_PID = os.getpid()
            _SESSION.close()

    return _SESSION


class NetFileError(OSError):
    """An error occurred while reading a remote file."""

    def __init__(self, message, url=None, status_code=None):
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class NetFileNotFoundError(NetFileError, FileNotFoundError):
    """A remote file does not exist."""


class NetFilePermissionError(NetFileError, PermissionError):
    """The client does not have permission to access a remote file."""


class RangeTransfer:
    """
    A single HTTP transfer representing a range of bytes.

    Parameters
    ----------
    url : str
        URL of the remote file.
    start : int, optional
        Start of the byte range to download, as an offset from the
        beginning of the file (inclusive, 0-based.)
    end : int or None
        End of the byte range to download, as an offset from the
        beginning of the file (exclusive, 0-based.)  If None, request
        all data until the end of the file.

    Attributes
    ----------
    request_url : str
        Original URL that was requested.
    response_url : str
        URL that was actually retrieved (after following redirections.)
    is_complete : bool
        True if the response contains the entire file; False if the
        response contains a byte range.
    file_size : int or None
        Total size of the remote file.  This may be None if the length
        is unknown.

    Notes
    -----
    The start and end parameters are requests that the server may or
    may not honor.  After creating a RangeTransfer object, call
    content() or iter_chunks() to retrieve the actual response data,
    which may be a subset or a superset of the requested range.

    """

    def __init__(self, url, start, end):
        self.request_url = url

        if start == 0 and end is None:
            method = "GET"
            headers = {}
        elif end is None:
            method = "GET"
            headers = {
                "Range": "bytes=%d-" % start,
                "Accept-Encoding": None,
            }
        elif end > start:
            method = "GET"
            headers = {
                "Range": "bytes=%d-%d" % (start, end - 1),
                "Accept-Encoding": None,
            }
        else:
            method = "HEAD"
            headers = {
                "Accept-Encoding": None,
            }

        session = _get_session()
        self._response = session.request(
            method, url, headers=headers, stream=True
        )
        self._content_iter = self._response.iter_content(4096)
        try:
            self._parse_headers(method, self._response)
        except Exception:
            self.close()
            raise

    def _parse_headers(self, method, response):
        """
        Parse the headers of the response object.

        Parameters
        ----------
        method : str
            The HTTP method used for the request.
        response : requests.Response
            The resulting response object.

        Returns
        -------
        N/A

        Notes
        -----
        - response_url is set to the URL of the response
        - file_size is set to the total file size
        - is_complete is set to true if the response is complete
        - _current_pos is set to the starting position
        - _expected_end_pos is set to the expected end position

        """
        self.response_url = response.url
        self.file_size = None
        self.is_complete = False
        self._current_pos = 0
        self._expected_end_pos = None

        # Raise an exception if an error occurs.
        if response.status_code >= 400 and response.status_code != 416:
            _LOGGER.info(
                "%s %s: %s", method, response.url, response.status_code
            )
            if response.status_code in (401, 403):
                cls = NetFilePermissionError
            elif response.status_code == 404:
                cls = NetFileNotFoundError
            else:
                cls = NetFileError
            raise cls(
                "%s Error: %s for url: %s"
                % (response.status_code, response.reason, response.url),
                url=response.url,
                status_code=response.status_code,
            )

        # Parse the Content-Range if this is a partial response.
        elif response.status_code in (206, 416):
            content_range = response.headers.get("Content-Range")
            if content_range:
                match = _CONTENT_RANGE_PATTERN.fullmatch(content_range)
                if not match:
                    raise NetFileError(
                        "Invalid Content-Range: %s" % content_range,
                        url=response.url,
                    )
                if match.group(1):
                    self._current_pos = int(match.group(1))
                    self._expected_end_pos = int(match.group(2)) + 1
                if match.group(3):
                    self.file_size = int(match.group(3))
            elif response.status_code == 206:
                raise NetFileError(
                    "Missing Content-Range in partial response",
                    url=response.url,
                )

        # Parse the Content-Length if this is a complete and
        # uncompressed response.
        elif 200 <= response.status_code < 300:
            self.is_complete = True
            content_encoding = response.headers.get("Content-Encoding")
            content_length = response.headers.get("Content-Length")
            if content_length and not content_encoding:
                try:
                    self.file_size = int(content_length)
                    self._expected_end_pos = self.file_size
                except ValueError:
                    raise NetFileError(
                        "Invalid Content-Length: %s" % content_length,
                        url=response.url,
                    )

        _LOGGER.info(
            "%s %s: %s %s-%s/%s",
            method,
            response.url,
            response.status_code,
            self._current_pos,
            self._expected_end_pos,
            self.file_size,
        )

        # If the response is an error (or an unhandled redirection)
        # then discard the body.
        if response.status_code >= 300:
            self.close()

    def close(self):
        """
        Finish reading data from the response.

        Any leftover data in the response body will be discarded and
        the underlying HTTP connection will be returned to the pool.

        Parameters
        ----------
        N/A

        Returns
        -------
        N/A

        """
        try:
            for data in self._content_iter:
                pass
        except Exception:
            pass
        self._response.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When exiting with a normal exception, shut down cleanly by
        # reading leftover response data.  When exiting abnormally
        # (SystemExit, KeyboardInterrupt), do nothing.
        if not exc_type or issubclass(exc_type, Exception):
            self.close()

    def __del__(self):
        # If the object is deleted without calling close(), forcibly
        # close the existing connection.
        response = getattr(self, "_response", None)
        if response:
            response.close()

    def iter_chunks(self):
        """
        Iterate over the response body as a sequence of chunks.

        Parameters
        ----------
        N/A

        Yields
        ------
        chunk_start : int
            Byte offset within the remote file corresponding to the
            start of this chunk.
        chunk_data : bytes
            Contents of the chunk.

        """
        for chunk_data in self._content_iter:
            chunk_start = self._current_pos
            self._current_pos += len(chunk_data)
            yield chunk_start, chunk_data
        if self.is_complete:
            self.file_size = self._current_pos

    def content(self):
        """
        Read the complete response.

        Parameters
        ----------
        N/A

        Returns
        -------
        start : int
            Byte offset within the remote file corresponding to the
            start of the response.
        data : bytes
            Contents of the response.

        """
        start = self._current_pos
        chunks = []
        for _, chunk_data in self.iter_chunks():
            chunks.append(chunk_data)
        return start, b"".join(chunks)


class NetFile(io.BufferedIOBase):
    """
    File object providing random access to a remote file over HTTP.

    Attributes
    ----------
    url : str
        URL of the remote file.
    buffering : int, optional
        Buffering policy.  If buffering = 0, internal buffering is
        disabled; each operation on the stream requires a separate
        request to the server.  If buffering = -2, the entire file is
        downloaded in a single request.  If buffering > 0, it
        specifies the minimum size of the internal buffer.  If
        buffering = -1, the default buffer size is used.

    """

    def __init__(self, url, buffering=-1):
        self.url = url
        self.name = url
        self.buffering = buffering
        self._pos = 0
        self._file_size = None
        self._buffer = b""
        self._buffer_start = 0
        self._buffer_end = 0
        self._current_url = self.url

    def _read_buffered_range(self, start, end):
        """
        Read a range of bytes from the internal buffer.

        Parameters
        ----------
        start : int
            Starting byte offset of the desired range.
        end : int
            Ending byte offset of the desired range.

        Returns
        -------
        data : memoryview
            A memoryview of the given byte range.

        """
        bstart = start - self._buffer_start
        bend = end - self._buffer_start
        if 0 <= bstart <= bend:
            return memoryview(self._buffer)[bstart:bend]
        else:
            return memoryview(b"")

    def _read_range(self, start, end):
        """
        Read a range of bytes from the remote file.

        The result is returned as a sequence of chunks; the sizes of
        the individual chunks are unspecified.  The total size may be
        less than requested if the end of the file is reached.

        Parameters
        ----------
        start : int
            Starting byte offset of the desired range.
        end : int or None
            Ending byte offset of the desired range, or None to read
            all data up to the end of the file.

        Yields
        ------
        data : memoryview
            A memoryview containing a chunk of the desired range.

        """
        # Read buffered data if available.
        if self._buffer_start <= start < self._buffer_end:
            if end is None:
                range_end = self._buffer_end
            else:
                range_end = min(end, self._buffer_end)
            yield self._read_buffered_range(start, range_end)
            start = range_end

        if end is not None and start >= end:
            return
        if self._file_size is not None and start >= self._file_size:
            return

        buffer_store = False

        if self.buffering == BUFFER_WHOLE_FILE:
            # Request entire file and save it in the internal buffer.
            req_start = 0
            req_end = None
            buffer_store = True
        elif end is None:
            # Request range from start to EOF and don't save it in the
            # buffer (since the result will be immediately consumed.)
            req_start = start
            req_end = None
        else:
            # Request a fixed range of bytes.  Save it in the buffer
            # if it is smaller than the maximum buffer size.
            buffer_size = self.buffering
            if buffer_size < 0:
                buffer_size = DEFAULT_BUFFER_SIZE
            req_start = start
            req_end = end
            if req_end < req_start + buffer_size:
                req_end = req_start + buffer_size
                buffer_store = True

        with RangeTransfer(self._current_url, req_start, req_end) as xfer:
            # Update current file URL.
            self._current_url = xfer.response_url

            # If we requested a range but the server doesn't support
            # random access, then unless buffering is disabled, save
            # entire file in the buffer.
            if self.buffering == 0:
                buffer_store = False
            elif xfer.is_complete and (start, end) != (0, None):
                buffer_store = True

            if buffer_store:
                # Load data into buffer and then return a copy to the
                # caller.
                (start, data) = xfer.content()
                self._buffer = data
                self._buffer_start = start
                self._buffer_end = start + len(data)
                if end is None:
                    end = self._buffer_end
                yield self._read_buffered_range(start, end)
            else:
                # Return requested data to caller without buffering.
                for chunk_start, chunk_data in xfer.iter_chunks():
                    rel_start = start - chunk_start
                    if 0 <= rel_start < len(chunk_data):
                        if end is None:
                            rel_end = len(chunk_data)
                        else:
                            rel_end = min(end - chunk_start, len(chunk_data))
                        yield memoryview(chunk_data)[rel_start:rel_end]
                        start = chunk_start + rel_end

            # Update file size.
            if self.buffering != 0:
                self._file_size = xfer.file_size

    def _get_size(self):
        """
        Determine the size of the remote file.

        Parameters
        ----------
        N/A

        Returns
        -------
        size : int or None
             Size of the remote file, if known.

        """
        size = self._file_size
        if size is None:
            if self.buffering == BUFFER_WHOLE_FILE:
                for _ in self._read_range(0, None):
                    pass
            else:
                with RangeTransfer(self._current_url, 0, 0) as xfer:
                    self._current_url = xfer.response_url
                    self._file_size = xfer.file_size
            size = self._file_size
            if self.buffering == 0:
                self._file_size = None
        return size

    def readable(self):
        """
        Determine whether the file supports read() and read1() operations.

        Parameters
        ----------
        N/A

        Returns
        -------
        True

        """
        return True

    def read(self, size=-1):
        """
        Read bytes from the file.

        Parameters
        ----------
        size : int
            Number of bytes to read, or -1 to read as many bytes as
            possible.

        Returns
        -------
        data : bytes
            Bytes retrieved from the file.  When the end of the file
            is reached, the length will be less than the requested
            size.

        """
        start = self._pos
        if size in (-1, None):
            end = None
        elif size >= 0:
            end = start + size
        else:
            raise ValueError("invalid size: %r" % (size,))

        result = b"".join(self._read_range(start, end))
        self._pos += len(result)
        return result

    def read1(self, size=-1):
        """
        Read bytes from the file.

        Parameters
        ----------
        size : int
            Maximum number of bytes to read, or -1 to read as many
            bytes as possible.

        Returns
        -------
        data : bytes
            Bytes retrieved from the file.  When the end of the file
            is reached, the length will be zero.

        """
        return self.read(size)

    def readinto(self, b):
        """
        Read bytes from the file.

        Parameters
        ----------
        b : writable bytes-like object
            Buffer in which to store the retrieved bytes.

        Returns
        -------
        count : int
            Number of bytes retrieved from the file and stored in b.
            When the end of the file is reached, the count will be
            less than the requested size.

        """
        b = memoryview(b).cast("B")
        start = self._pos
        end = start + len(b)
        count = 0
        for chunk in self._read_range(start, end):
            b[count : count + len(chunk)] = chunk
            count += len(chunk)
        self._pos += count
        return count

    def readinto1(self, b):
        """
        Read bytes from the file.

        Parameters
        ----------
        b : writable bytes-like object
            Buffer in which to store the retrieved bytes.

        Returns
        -------
        count : int
            Number of bytes retrieved from the file and stored in b.
            When the end of the file is reached, the count will be
            zero.

        """
        return self.readinto(b)

    def seekable(self):
        """
        Determine whether the file supports seek() and tell() operations.

        Parameters
        ----------
        N/A

        Returns
        -------
        True

        """
        return True

    def seek(self, offset, whence=os.SEEK_SET):
        """
        Set the current file position.

        Parameters
        ----------
        offset : int
            Byte offset of the new file position, relative to the base
            position specified by whence.
        whence : int, optional
            SEEK_SET (0, default) if offset is relative to the start
            of the file; SEEK_CUR (1) if offset is relative to the
            current file position; SEEK_END (2) if offset is relative
            to the end of the file.

        Returns
        -------
        offset : int
            Byte offset of the new file position.

        """
        if whence == os.SEEK_SET:
            pos = offset
        elif whence == os.SEEK_CUR:
            pos = self._pos + offset
        elif whence == os.SEEK_END:
            size = self._get_size()
            if size is None:
                raise NetFileError(
                    "size of remote file is unknown", url=self._current_url
                )
            pos = size + offset
        else:
            raise ValueError("invalid whence: %r" % (whence,))
        if pos < 0:
            raise ValueError("pos < 0")
        self._pos = pos
        return pos

    def tell(self):
        """
        Retrieve the current file position.

        Parameters
        ----------
        N/A

        Returns
        -------
        offset : int
            Byte offset of the current file position.

        """
        return self._pos


def openurl(
    url,
    mode="r",
    *,
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
    check_access=False,
):
    """
    Open a URL as a random-access file object.

    Parameters
    ----------
    url : str
        URL of the remote file.
    mode : str, optional
        Whether to access the file in text mode ('r' or 'rt';
        default), or binary mode ('rb').
    buffering : int, optional
        Buffering policy.  If buffering = 0, internal buffering is
        disabled; each operation on the stream requires a separate
        request to the server.  If buffering = -2, the entire file is
        downloaded in a single request.  If buffering > 0, it
        specifies the minimum size of the internal buffer.  If
        buffering = -1, the default buffer size is used.
    encoding : str, optional
        Name of character encoding used in text mode.
    errors : str, optional
        Error handling strategy used for invalid byte sequences in
        text mode.  See the documentation of the standard "open"
        function for details.
    newline : str, optional
        Newline translation mode used in text mode.  See the
        documentation of the standard "open" function for details.
    check_access : bool, optional
        If true, raise an exception immediately if the file does not
        exist or is not accessible.  If false (default), no exception
        is raised until the first time you call read() or a related
        function.

    Returns
    -------
    nf : io.IOBase
        A file object, implementing either the binary file API
        (io.BufferedIOBase) or text file API (io.TextIOBase).

    """
    (scheme, netloc, path, _, _, _) = urllib.parse.urlparse(url)
    if scheme == "":
        raise NetFileError("no scheme specified for URL: %r" % (url,), url=url)

    if scheme == "file":
        if netloc.lower() not in ("", "localhost"):
            raise NetFileError("invalid file URL: %r" % (url,))
        local_path = urllib.request.url2pathname(path)
        return open(
            local_path,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    nf = NetFile(url, buffering=buffering)

    if check_access:
        nf._get_size()

    if mode == "rb":
        return nf
    elif mode == "r" or mode == "rt":
        return io.TextIOWrapper(
            nf, encoding=encoding, errors=errors, newline=newline
        )
    else:
        return ValueError("invalid mode: %r" % (mode,))
