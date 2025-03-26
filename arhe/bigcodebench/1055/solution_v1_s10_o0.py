
def task_func(colors, states):
    num_cols = min(len(colors), len(states))
    combinations = ['{}:{}'.format(color, state) for color, state in itertools.product(colors, states)]
    random.shuffle(combinations)
    
    df = pd.DataFrame(columns=[f'Column_{i+1}' for i in range(num_cols)])
    for i, comb in enumerate(combinations):
        col_idx = i % num_cols
        df.loc[i, f'Column_{col_idx+1}'] = comb
    
    df.fillna("", inplace=True)
    
    return df

import unittest
import pandas as pd
import random
class TestCases(unittest.TestCase):
    """Test cases for task_func."""
    def test_empty_lists(self):
        """Test with empty color and state lists."""
        self.assertEqual(task_func([], []).empty, True)
    def test_single_color_and_state(self):
        """Test with one color and one state."""
        random.seed(0)
        result = task_func(["Red"], ["Solid"])
        expected = pd.DataFrame({"Color:State 1": ["Red:Solid"]})
        pd.testing.assert_frame_equal(result, expected)
    def test_multiple_colors_single_state(self):
        """Test with multiple colors and a single state."""
        random.seed(1)
        result = task_func(["Red", "Blue", "Green"], ["Solid"])
        expected_combinations = set(["Red:Solid", "Blue:Solid", "Green:Solid"])
        result_combinations = set(result["Color:State 1"])
        self.assertEqual(result_combinations, expected_combinations)
    def test_single_color_multiple_states(self):
        """Test with a single color and multiple states."""
        random.seed(2)
        result = task_func(["Red"], ["Solid", "Liquid", "Gas"])
        expected_combinations = set(["Red:Solid", "Red:Liquid", "Red:Gas"])
        result_combinations = set(result["Color:State 1"])
        self.assertEqual(result_combinations, expected_combinations)
    def test_multiple_colors_and_states(self):
        """Test with multiple colors and states."""
        random.seed(3)
        colors = ["Red", "Blue"]
        states = ["Solid", "Liquid"]
        result = task_func(colors, states)
        expected_combinations = set(
            [f"{color}:{state}" for color in colors for state in states]
        )
        result_combinations = set(result.values.flatten())
        self.assertEqual(result_combinations, expected_combinations)
testcases = TestCases()
testcases.test_empty_lists()
