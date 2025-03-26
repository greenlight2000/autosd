
def task_func(rows=100, columns=3):
    import pandas as pd
    import numpy as np
    data = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=(rows, columns))
    col_names = [chr(ord('a') + i) for i in range(columns)]
    df = pd.DataFrame(data, columns=col_names)
    return df

import unittest
import numpy as np
class TestCases(unittest.TestCase):
    """Tests case for function `task_func`."""
    def test_dataframe_shape_default(self):
        """Test if the DataFrame has default shape (100 rows, 3 columns) with default parameters."""
        np.random.seed(1)
        df_test = task_func()
        self.assertEqual(df_test.shape, (100, 3))
    def test_dataframe_shape_custom_rows(self):
        """Test if the DataFrame has the correct shape when a custom number of rows is specified."""
        np.random.seed(2)
        df_test = task_func(50)
        self.assertEqual(df_test.shape, (50, 3))
    def test_dataframe_shape_custom_columns(self):
        """Test if the DataFrame has the correct shape with a custom number of columns."""
        np.random.seed(3)
        df_test = task_func(50, 5)
        self.assertEqual(df_test.shape, (50, 5))
    def test_dataframe_columns_default(self):
        """Test if the DataFrame has default column names ['a', 'b', 'c'] with default parameters."""
        np.random.seed(4)
        df_test = task_func()
        self.assertListEqual(list(df_test.columns), ["a", "b", "c"])
    def test_dataframe_columns_custom(self):
        """Test if the DataFrame has the correct column names when a custom number of columns is specified."""
        np.random.seed(5)
        df_test = task_func(columns=5)
        expected_columns = ["a", "b", "c", "d", "e"]
        self.assertListEqual(list(df_test.columns), expected_columns)
    def test_dataframe_values(self):
        """Test if each cell in the DataFrame contains a letter from the English alphabet."""
        np.random.seed(6)
        df_test = task_func()
        for col in df_test.columns:
            self.assertTrue(
                set(df_test[col].unique()).issubset(set("abcdefghijklmnopqrstuvwxyz"))
            )
    def test_dataframe_empty(self):
        """Test if an empty DataFrame is created when 0 rows are specified."""
        np.random.seed(7)
        df_test = task_func(0)
        self.assertEqual(df_test.shape, (0, 3))
testcases = TestCases()
# code status: pass