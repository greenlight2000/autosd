
def task_func(data_dict):
    df = pd.DataFrame(data_dict.items(), columns=['Key', 'Value'])
    df = df.dropna()
    
    if df.empty or df['Value'].nunique() == 1:
        return pd.DataFrame(columns=['Key', 'Value']), None
    
    min_val = df['Value'].min()
    max_val = df['Value'].max()
    
    num_bins = min(max(2, len(df) // 2), 11)
    
    plot = sns.histplot(df['Value'], bins=num_bins, kde=False)
    plot.set_title(PLOT_TITLE)
    
    return df, plot

import unittest
import pandas as pd
class TestCases(unittest.TestCase):
    """Test cases for function task_func."""
    def test_dataframe_creation(self):
        """
        Test if the function correctly creates a DataFrame from the input dictionary.
        """
        data = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
        df, _ = task_func(data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (4, 2))
    def test_distribution_plot(self):
        """
        Test if the function correctly creates a distribution plot with the correct title and non-empty bars.
        """
        data = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
        _, plot = task_func(data)
        self.assertEqual(plot.get_title(), "Value Distribution")
        self.assertTrue(len(plot.patches) > 0)
    def test_empty_dictionary(self):
        """
        Test if the function correctly handles an empty dictionary, returning an empty DataFrame and no plot.
        """
        data = {}
        df, plot = task_func(data)
        self.assertEqual(df.shape, (0, 0))
        self.assertIsNone(plot)
    def test_number_of_bins(self):
        """
        Test if the function dynamically calculates the number of bins for the plot based on the data.
        """
        data = {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        _, plot = task_func(data)
        self.assertTrue(len(plot.patches) <= 11)
    def test_dataframe_without_none(self):
        """
        Test if the function correctly removes rows with None values from the DataFrame.
        """
        data = {"a": [1, 2, None, 4], "b": [5, None, 7, 8]}
        df, _ = task_func(data)
        self.assertEqual(df.shape, (2, 2))
        self.assertNotIn(None, df.values.flatten())
testcases = TestCases()
testcases.test_empty_dictionary()
