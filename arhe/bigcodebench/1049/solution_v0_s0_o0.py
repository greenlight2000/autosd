import re
import pandas as pd
def task_func(input_string: str) -> pd.DataFrame:

    pass
import re
import pandas as pd
def task_func(input_string: str) -> pd.DataFrame:
    lines = input_string.split('\n')
    cleaned_lines = [re.sub(r'\t', ' ', line.strip()) for line in lines if line.strip()]
    df = pd.DataFrame({'Text': cleaned_lines})
    return df
import pandas as pd
import unittest
class TestCases(unittest.TestCase):
    """Tests for task_func."""
    def test_basic_string(self):
        """
        Test with a basic multi-line string.
        """
        input_str = "line1\nline2 with a\ttab\nline3"
        expected_output = pd.DataFrame({"Text": ["line1", "line2 with a tab", "line3"]})
        pd.testing.assert_frame_equal(task_func(input_str), expected_output)
    def test_empty_string(self):
        """
        Test with an empty string.
        """
        input_str = ""
        expected_output = pd.DataFrame(columns=["Text"])
        pd.testing.assert_frame_equal(task_func(input_str), expected_output)
    def test_string_with_empty_lines(self):
        """
        Test with a string that contains empty lines.
        """
        input_str = "line1\n\nline3"
        expected_output = pd.DataFrame({"Text": ["line1", "line3"]})
        pd.testing.assert_frame_equal(task_func(input_str), expected_output)
    def test_string_with_only_tabs(self):
        """
        Test with a string that contains only tabs.
        """
        input_str = "\t\t\t"
        expected_output = pd.DataFrame(columns=["Text"])
        pd.testing.assert_frame_equal(task_func(input_str), expected_output)
    def test_string_with_mixed_whitespace(self):
        """
        Test with a string that contains a mix of tabs and spaces.
        """
        input_str = "line1\n \t \nline3"
        expected_output = pd.DataFrame({"Text": ["line1", "line3"]})
        pd.testing.assert_frame_equal(task_func(input_str), expected_output)
testcases = TestCases()
testcases.test_empty_string()
