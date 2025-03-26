import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
def task_func(csv_file_path, target_column="target", test_size=0.2, n_estimators=100):

    pass
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
def task_func(csv_file_path, target_column="target", test_size=0.2, n_estimators=100):
    try:
        data = pd.read_csv(csv_file_path)
        if target_column not in data.columns:
            raise ValueError(f"Specified target column '{target_column}' not found in the CSV file.")
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators)
        rf_classifier.fit(X_train, y_train)
        
        y_pred = rf_classifier.predict(X_test)
        
        report = classification_report(y_test, y_pred)
        
        return report
    
    except Exception as e:
        return str(e)
csv_file_path = "data.csv"
target_column = "class"
import unittest
from unittest.mock import patch
import pandas as pd
class TestCases(unittest.TestCase):
    """Test cases for task_func."""
    @patch("pandas.read_csv")
    def test_default_parameters(self, mock_read_csv):
        """
        Test task_func with default parameters using an adequately sized mock dataset.
        """
        mock_data = {
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [0, 1] * 50,  # Alternating 0s and 1s
        }
        mock_read_csv.return_value = pd.DataFrame(mock_data)
        result = task_func("dummy_path.csv")
        self.assertIn("precision", result)
    @patch("pandas.read_csv")
    def test_non_default_target_column(self, mock_read_csv):
        """
        Test task_func with a non-default target column using a larger mock dataset.
        """
        mock_data = {
            "feature1": range(100),
            "feature2": range(100, 200),
            "label": [1, 0] * 50,  # Alternating 1s and 0s
        }
        mock_read_csv.return_value = pd.DataFrame(mock_data)
        result = task_func("dummy_path.csv", target_column="label")
        self.assertIn("precision", result)
    @patch("pandas.read_csv")
    def test_different_test_size(self, mock_read_csv):
        """
        Test task_func with a different test size and a larger dataset.
        """
        mock_data = {
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [0, 1, 1, 0] * 25,  # Repeated pattern
        }
        mock_read_csv.return_value = pd.DataFrame(mock_data)
        result = task_func("dummy_path.csv", test_size=0.5)
        self.assertIn("precision", result)
    @patch("pandas.read_csv")
    def test_different_n_estimators(self, mock_read_csv):
        """
        Test task_func with a different number of estimators and an expanded dataset.
        """
        mock_data = {
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [1, 0] * 50,  # Alternating 1s and 0s
        }
        mock_read_csv.return_value = pd.DataFrame(mock_data)
        result = task_func("dummy_path.csv", n_estimators=50)
        self.assertIn("precision", result)
    @patch("pandas.read_csv")
    def test_missing_target_column(self, mock_read_csv):
        """
        Test task_func with a missing target column.
        """
        mock_read_csv.return_value = pd.DataFrame(
            {"feature1": [1, 2], "feature2": [3, 4]}
        )
        with self.assertRaises(ValueError):
            task_func("dummy_path.csv", target_column="not_exist")
testcases = TestCases()
testcases.test_missing_target_column()
