
def task_func(xml_content, output_csv_path):
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ET.ParseError(f"Error parsing XML content: {str(e)}")

    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for child in root.iter():
                csvwriter.writerow([child.tag, child.text if child.text is not None else ""])
    except IOError as e:
        raise IOError(f"Error writing to CSV file: {str(e)}")

import unittest
import xml.etree.ElementTree as ET
import csv
import shutil
from pathlib import Path
import os
class TestCases(unittest.TestCase):
    """Test cases for task_func."""
    test_data_dir = "mnt/data/task_func_data"
    def setUp(self):
        """Set up method to create a directory for test files."""
        self.test_dir = Path(self.test_data_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
    def check_csv_content(self, xml_content, csv_path):
        """Helper function to check if the CSV content matches the XML content."""
        root = ET.fromstring(xml_content)
        expected_data = [
            [elem.tag, elem.text if elem.text is not None else ""]
            for elem in root.iter()
        ]
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            csv_data = list(reader)
        self.assertEqual(expected_data, csv_data)
    def test_simple_xml(self):
        """Test with simple XML content."""
        xml_content = "<root><element>data</element></root>"
        csv_output = self.test_dir / "output_scenario_0.csv"
        task_func(xml_content, csv_output)
        self.check_csv_content(xml_content, csv_output)
    def test_nested_xml(self):
        """Test with nested XML content."""
        xml_content = "<root><parent><child>data</child></parent></root>"
        csv_output = self.test_dir / "output_scenario_1.csv"
        task_func(xml_content, csv_output)
        self.check_csv_content(xml_content, csv_output)
    def test_empty_xml(self):
        """Test with an empty XML."""
        xml_content = "<root></root>"
        csv_output = self.test_dir / "output_scenario_2.csv"
        task_func(xml_content, csv_output)
        self.check_csv_content(xml_content, csv_output)
    def test_xml_with_attributes(self):
        """Test with an XML that contains elements with attributes."""
        xml_content = '<root><element attr="value">data</element></root>'
        csv_output = self.test_dir / "output_scenario_3.csv"
        task_func(xml_content, csv_output)
        self.check_csv_content(xml_content, csv_output)
    def test_large_xml(self):
        """Test with a larger XML file."""
        xml_content = (
            "<root>"
            + "".join([f"<element>{i}</element>" for i in range(100)])
            + "</root>"
        )
        csv_output = self.test_dir / "output_scenario_4.csv"
        task_func(xml_content, csv_output)
        self.check_csv_content(xml_content, csv_output)
    def test_invalid_xml_content(self):
        """Test with invalid XML content to trigger ET.ParseError."""
        xml_content = "<root><element>data</element"  # Malformed XML
        csv_output = self.test_dir / "output_invalid_xml.csv"
        with self.assertRaises(ET.ParseError):
            task_func(xml_content, csv_output)
    def test_unwritable_csv_path(self):
        """Test with an unwritable CSV path to trigger IOError."""
        xml_content = "<root><element>data</element></root>"
        csv_output = self.test_dir / "non_existent_directory" / "output.csv"
        with self.assertRaises(IOError):
            task_func(xml_content, csv_output)
    def tearDown(self):
        # Cleanup the test directories
        dirs_to_remove = ["mnt/data", "mnt"]
        for dir_path in dirs_to_remove:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
testcases = TestCases()
testcases.setUp()
# code status: passtestcases.tearDown()
