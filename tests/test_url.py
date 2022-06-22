import gzip
import http.server
import threading
import unittest

import wfdb.io._url


class TestNetFiles(unittest.TestCase):
    """
    Test accessing remote files.
    """

    def test_requests(self):
        """
        Test reading a remote file using various APIs.

        This tests that we can create a file object using
        wfdb.io._url.openurl(), and tests that the object implements
        the standard Python API functions for a file of the
        appropriate type.

        Parameters
        ----------
        N/A

        Returns
        -------
        N/A

        """

        text_data = """
        BERNARDO: Who's there?
        FRANCISCO: Nay, answer me: stand, and unfold yourself.
        BERNARDO: Long live the king!
        FRANCISCO: Bernardo?
        BERNARDO: He.
        FRANCISCO: You come most carefully upon your hour.
        BERNARDO: 'Tis now struck twelve; get thee to bed, Francisco.
        """
        binary_data = text_data.encode()
        file_content = {"/foo.txt": binary_data}

        # Test all possible combinations of:
        #  - whether or not the server supports compression
        #  - whether or not the server supports random access
        #  - chosen buffering policy
        for allow_gzip in (False, True):
            for allow_range in (False, True):
                with DummyHTTPServer(
                    file_content=file_content,
                    allow_gzip=allow_gzip,
                    allow_range=allow_range,
                ) as server:
                    url = server.url("/foo.txt")
                    for buffering in (-2, -1, 0, 20):
                        self._test_text(url, text_data, buffering)
                        self._test_binary(url, binary_data, buffering)

    def _test_text(self, url, content, buffering):
        """
        Test reading a URL using text-mode file APIs.

        Parameters
        ----------
        url : str
            URL of the remote resource.
        content : str
            Expected content of the resource.
        buffering : int
            Buffering policy for openurl().

        Returns
        -------
        N/A

        """
        # read(-1), readable(), seekable()
        with wfdb.io._url.openurl(url, "r", buffering=buffering) as tf:
            self.assertTrue(tf.readable())
            self.assertTrue(tf.seekable())
            self.assertEqual(tf.read(), content)
            self.assertEqual(tf.read(), "")

        # read(10)
        with wfdb.io._url.openurl(url, "r", buffering=buffering) as tf:
            result = ""
            while True:
                chunk = tf.read(10)
                result += chunk
                if len(chunk) < 10:
                    break
            self.assertEqual(result, content)

        # readline(), seek(), tell()
        with wfdb.io._url.openurl(url, "r", buffering=buffering) as tf:
            result = ""
            while True:
                rpos = tf.tell()
                tf.seek(0)
                tf.seek(rpos)
                chunk = tf.readline()
                result += chunk
                if len(chunk) == 0:
                    break
            self.assertEqual(result, content)

    def _test_binary(self, url, content, buffering):
        """
        Test reading a URL using binary-mode file APIs.

        Parameters
        ----------
        url : str
            URL of the remote resource.
        content : bytes
            Expected content of the resource.
        buffering : int
            Buffering policy for openurl().

        Returns
        -------
        N/A

        """
        # read(-1), readable(), seekable()
        with wfdb.io._url.openurl(url, "rb", buffering=buffering) as bf:
            self.assertTrue(bf.readable())
            self.assertTrue(bf.seekable())
            self.assertEqual(bf.read(), content)
            self.assertEqual(bf.read(), b"")
            self.assertEqual(bf.tell(), len(content))

        # read(10)
        with wfdb.io._url.openurl(url, "rb", buffering=buffering) as bf:
            result = b""
            while True:
                chunk = bf.read(10)
                result += chunk
                if len(chunk) < 10:
                    break
            self.assertEqual(result, content)
            self.assertEqual(bf.tell(), len(content))

        # readline()
        with wfdb.io._url.openurl(url, "rb", buffering=buffering) as bf:
            result = b""
            while True:
                chunk = bf.readline()
                result += chunk
                if len(chunk) == 0:
                    break
            self.assertEqual(result, content)
            self.assertEqual(bf.tell(), len(content))

        # read1(10), seek(), tell()
        with wfdb.io._url.openurl(url, "rb", buffering=buffering) as bf:
            bf.seek(0, 2)
            self.assertEqual(bf.tell(), len(content))
            bf.seek(0)
            result = b""
            while True:
                rpos = bf.tell()
                bf.seek(0)
                bf.seek(rpos)
                chunk = bf.read1(10)
                result += chunk
                if len(chunk) == 0:
                    break
            self.assertEqual(result, content)
            self.assertEqual(bf.tell(), len(content))

        # readinto(bytearray(10))
        with wfdb.io._url.openurl(url, "rb", buffering=buffering) as bf:
            result = b""
            chunk = bytearray(10)
            while True:
                count = bf.readinto(chunk)
                result += chunk[:count]
                if count < 10:
                    break
            self.assertEqual(result, content)
            self.assertEqual(bf.tell(), len(content))

        # readinto1(bytearray(10))
        with wfdb.io._url.openurl(url, "rb", buffering=buffering) as bf:
            result = b""
            chunk = bytearray(10)
            while True:
                count = bf.readinto1(chunk)
                result += chunk[:count]
                if count == 0:
                    break
            self.assertEqual(result, content)
            self.assertEqual(bf.tell(), len(content))


class TestRemoteFLACFiles(unittest.TestCase):
    """
    Test reading FLAC files over HTTP.
    """

    def test_whole_file(self):
        """
        Test reading a complete FLAC file using local and HTTP APIs.

        This tests that we can read the file 'sample-data/flacformats.d2'
        (a 24-bit FLAC stream) using the soundfile library, first by
        reading the file from the local filesystem, and then using
        wfdb.io._url.openurl() to access it through a simulated web server.

        This is meant to verify that the soundfile library works using only
        the standard Python file object API (as implemented by
        wfdb.io._url.NetFile), and doesn't require the input file to be an
        actual io.FileIO object.

        Parameters
        ----------
        N/A

        Returns
        -------
        N/A

        """
        import soundfile
        import numpy as np

        data_file_path = "sample-data/flacformats.d2"
        expected_format = "FLAC"
        expected_subtype = "PCM_24"

        # Read the file using standard file I/O
        sf1 = soundfile.SoundFile(data_file_path)
        self.assertEqual(sf1.format, expected_format)
        self.assertEqual(sf1.subtype, expected_subtype)
        data1 = sf1.read()

        # Read the file using HTTP
        with open(data_file_path, "rb") as f:
            file_content = {"/foo.dat": f.read()}
        with DummyHTTPServer(file_content) as server:
            url = server.url("/foo.dat")
            file2 = wfdb.io._url.openurl(url, "rb")
            sf2 = soundfile.SoundFile(file2)
            self.assertEqual(sf2.format, expected_format)
            self.assertEqual(sf2.subtype, expected_subtype)
            data2 = sf2.read()

        # Check that results are equal
        np.testing.assert_array_equal(data1, data2)


class DummyHTTPServer(http.server.HTTPServer):
    """
    HTTPServer used to simulate a web server for testing.

    The server may be used as a context manager (using "with"); during
    execution of the "with" block, a background thread runs that
    listens for and handles client requests.

    Attributes
    ----------
    file_content : dict
        Dictionary containing the content of each file on the server.
        The keys are absolute paths (such as "/foo.txt"); the values
        are the corresponding content (bytes).
    allow_gzip : bool, optional
        True if the server should return compressed responses (using
        "Content-Encoding: gzip") when the client requests them (using
        "Accept-Encoding: gzip").
    allow_range : bool, optional
        True if the server should return partial responses (using 206
        Partial Content and "Content-Range") when the client requests
        them (using "Range").
    server_address : tuple (str, int), optional
        A tuple specifying the address and port number where the
        server should listen for connections.  If the port is 0, an
        arbitrary unused port is selected.  The default address is
        "127.0.0.1" and the default port is 0.

    """

    def __init__(
        self,
        file_content,
        allow_gzip=True,
        allow_range=True,
        server_address=("127.0.0.1", 0),
    ):
        super().__init__(server_address, DummyHTTPRequestHandler)
        self.file_content = file_content
        self.allow_gzip = allow_gzip
        self.allow_range = allow_range

    def url(self, path="/"):
        """
        Generate a URL that points to a file on this server.

        Parameters
        ----------
        path : str, optional
            Path of the file on the server.

        Returns
        -------
        url : str
            Absolute URL for the specified file.

        """
        return "http://127.0.0.1:%d/%s" % (
            self.server_address[1],
            path.lstrip("/"),
        )

    def __enter__(self):
        super().__enter__()
        self.thread = threading.Thread(target=self.serve_forever)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        self.thread.join()
        self.thread = None
        return super().__exit__(exc_type, exc_val, exc_tb)


class DummyHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    HTTPRequestHandler used to simulate a web server for testing.
    """

    def do_HEAD(self):
        self.send_head()

    def do_GET(self):
        body = self.send_head()
        self.wfile.write(body)

    def log_message(self, message, *args):
        pass

    def send_head(self):
        content = self.server.file_content.get(self.path)
        if content is None:
            self.send_error(404)
            return b""

        headers = {"Content-Type": "text/plain"}
        status = 200

        if self.server.allow_gzip:
            headers["Vary"] = "Accept-Encoding"
            if "gzip" in self.headers.get("Accept-Encoding", ""):
                content = gzip.compress(content)
                headers["Content-Encoding"] = "gzip"

        if self.server.allow_range:
            headers["Accept-Ranges"] = "bytes"
            req_range = self.headers.get("Range", "")
            if req_range.startswith("bytes="):
                start, end = req_range.split("=")[1].split("-")
                start = int(start)
                if end == "":
                    end = len(content)
                else:
                    end = min(len(content), int(end) + 1)
                if start < end:
                    status = 206
                    resp_range = "bytes %d-%d/%d" % (
                        start,
                        end - 1,
                        len(content),
                    )
                    content = content[start:end]
                else:
                    status = 416
                    resp_range = "bytes */%d" % len(content)
                    content = b""
                headers["Content-Range"] = resp_range

        headers["Content-Length"] = len(content)
        self.send_response(status)
        for h, v in sorted(headers.items()):
            self.send_header(h, v)
        self.end_headers()
        return content


if __name__ == "__main__":
    unittest.main()
