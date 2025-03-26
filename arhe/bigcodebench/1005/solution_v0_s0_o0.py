import urllib.request
import zipfile
import os
import urllib.error
def task_func(
    url: str,
    save_path: str = "downloaded_file.zip",
    extract_path: str = "extracted_files",
) -> str:

    pass
import urllib.request
import zipfile
import os
import urllib.error
def task_func(
    url: str,
    save_path: str = "downloaded_file.zip",
    extract_path: str = "extracted_files",
) -> str:
    try:
        # Download the ZIP file from the specified URL
        urllib.request.urlretrieve(url, save_path)
        
        # Extract the contents of the ZIP file
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Delete the downloaded ZIP file
        os.remove(save_path)
        
        return os.path.abspath(extract_path)
    
    except urllib.error.URLError as e:
        return f"URL Error: {str(e)}"
    
    except zipfile.BadZipFile:
        return "Error: Corrupted ZIP file"
    
    except FileNotFoundError:
        return "Error: File not found"
    
    except Exception as e:
        return f"Error: {str(e)}"
url = "https://example.com/example.zip"
import unittest
import os
import urllib.error
import shutil
from pathlib import Path
class TestCases(unittest.TestCase):
    """Test cases for the task_func function."""
    base_path = "mnt/data/task_func_data"
    def setUp(self):
        # Ensure the base path is absolute
        self.base_path = os.path.abspath(self.base_path)
        # Create base directory for test data
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
    def test_successful_download_and_extraction_sample_1(self):
        """Test Case 1: Successful Download and Extraction of Sample 1"""
        url = "https://getsamplefiles.com/download/zip/sample-1.zip"
        save_path = Path(self.base_path) / "sample_1_download.zip"
        extract_path = Path(self.base_path) / "sample_1_extract"
        result_path = task_func(url, save_path, extract_path)
        self.assertEqual(result_path, extract_path)
        self.assertTrue(os.path.exists(extract_path))
        self.assertFalse(os.path.exists(save_path))
    def test_successful_download_and_extraction_sample_2(self):
        """Test Case 2: Successful Download and Extraction of Sample 2"""
        url = "https://getsamplefiles.com/download/zip/sample-2.zip"
        save_path = Path(self.base_path) / "sample_2_download.zip"
        extract_path = Path(self.base_path) / "sample_2_extract"
        result_path = task_func(url, save_path, extract_path)
        self.assertEqual(result_path, extract_path)
        self.assertTrue(os.path.exists(extract_path))
        self.assertFalse(os.path.exists(save_path))
    def test_invalid_url(self):
        """Test Case 3: Invalid URL"""
        url = "https://invalidurl.com/nonexistent.zip"
        save_path = Path(self.base_path) / "invalid_url.zip"
        extract_path = Path(self.base_path) / "invalid_url_extract"
        result = task_func(url, save_path, extract_path)
        self.assertTrue(result.startswith("URL Error:"))
    def test_file_already_exists_at_save_path(self):
        """Test Case 4: File Already Exists at Save Path"""
        url = "https://getsamplefiles.com/download/zip/sample-1.zip"
        save_path = Path(self.base_path) / "existing_file.zip"
        extract_path = Path(self.base_path) / "existing_file_extract"
        # Create a dummy file at the save path
        with open(save_path, "w") as file:
            file.write("Dummy content")
        result_path = task_func(url, save_path, extract_path)
        self.assertEqual(result_path, extract_path)
        self.assertFalse(os.path.exists(save_path))
    def test_extraction_path_already_exists(self):
        """Test Case 5: Extraction Path Already Exists"""
        url = "https://getsamplefiles.com/download/zip/sample-2.zip"
        save_path = Path(self.base_path) / "extract_path_exists.zip"
        extract_path = Path(self.base_path) / "existing_extract_path"
        # Create the extraction path directory
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        result_path = task_func(url, save_path, extract_path)
        self.assertEqual(result_path, extract_path)
    def tearDown(self):
        # Clean up any files or directories created during the tests
        shutil.rmtree(self.base_path, ignore_errors=True)
        # Cleanup the test directories
        dirs_to_remove = ["mnt/data", "mnt"]
        for dir_path in dirs_to_remove:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
testcases = TestCases()
testcases.setUp()
testcases.test_extraction_path_already_exists()
testcases.tearDown()
