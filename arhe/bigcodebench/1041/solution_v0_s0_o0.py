import re
import os
def task_func(request):

    pass
import re
import os
def task_func(request):
    # Extract filename from the request
    match = re.search(r'GET /(.*) HTTP', request)
    if match:
        filename = match.group(1)
    else:
        return "400 Bad Request: Invalid HTTP request"

    # Check if the file exists
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            content = file.read()
            response = f"200 OK\nContent-Length: {len(content)}\n\n{content}"
    else:
        response = "404 Not Found: File not found"

    return response
import unittest
import re
import os
from unittest.mock import mock_open, patch
class TestCases(unittest.TestCase):
    """Test cases for the task_func function."""
    def setUp(self):
        """Set up the environment for testing by creating test files."""
        with open("test.txt", "w", encoding="utf-8") as f:
            f.write("This is a test file.")
    def tearDown(self):
        """Clean up the environment by deleting the test files created."""
        os.remove("test.txt")
    def test_file_found(self):
        """Test the response when the requested file is found."""
        request = "GET /test.txt HTTP/1.1"
        expected_response = (
            "HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\nThis is a test file."
        )
        self.assertEqual(task_func(request), expected_response)
    def test_file_not_found(self):
        """Test the response when the requested file is not found."""
        request = "GET /nonexistent.txt HTTP/1.1"
        expected_response = "HTTP/1.1 404 NOT FOUND\r\n\r\nFile Not Found"
        self.assertEqual(task_func(request), expected_response)
    def test_bad_request(self):
        """Test the response for a badly formatted request."""
        request = "BAD REQUEST"
        expected_response = "HTTP/1.1 400 BAD REQUEST\r\n\r\nBad Request"
        self.assertEqual(task_func(request), expected_response)
    def test_empty_request(self):
        """Test the response for an empty request."""
        request = ""
        expected_response = "HTTP/1.1 400 BAD REQUEST\r\n\r\nBad Request"
        self.assertEqual(task_func(request), expected_response)
    def test_invalid_method_request(self):
        """Test the response for a request with an invalid HTTP method."""
        request = "POST /test.txt HTTP/1.1"
        expected_response = "HTTP/1.1 400 BAD REQUEST\r\n\r\nBad Request"
        self.assertEqual(task_func(request), expected_response)
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    def test_internal_server_error(self, mock_file):
        """Test the response when there's an internal server error (e.g., file read error)."""
        mock_file.side_effect = Exception("Mocked exception")
        request = "GET /test.txt HTTP/1.1"
        expected_response = (
            "HTTP/1.1 500 INTERNAL SERVER ERROR\r\n\r\nInternal Server Error"
        )
        self.assertEqual(task_func(request), expected_response)
testcases = TestCases()
testcases.setUp()
testcases.test_bad_request()
testcases.tearDown()
