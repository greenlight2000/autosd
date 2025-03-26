import numpy as np
import random
import matplotlib.pyplot as plt
# Constants
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
NUMBERS = list(range(1, 27))
def task_func(n_pairs=26):

    pass
def task_func(n_pairs=26):
    if n_pairs < 1 or n_pairs > 26:
        raise ValueError("'n_pairs' must be within the range of 1 to 26, inclusive.")

    pairs = [(letter, number) for letter in LETTERS for number in NUMBERS]
    random.shuffle(pairs)
    counts = [random.randint(1, 9) for _ in range(n_pairs)]

    pairs = pairs[:n_pairs]
    counts = counts[:n_pairs]

    labels = [f"{pair[0]}:{pair[1]}" for pair in pairs]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(n_pairs), counts, tick_label=labels)
    plt.title("Random Letter:Number Pairs Chart")
    plt.xlabel("Letter:Number Pairs")
    plt.ylabel("Counts")

    return bars
import unittest
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import random
class TestCases(unittest.TestCase):
    """Tests for the function task_func."""
    def test_return_type(self):
        """Verify the returned type of the function."""
        random.seed(0)
        ax = task_func(5)
        self.assertIsInstance(
            ax, BarContainer, "The returned object is not of the expected type."
        )
    def test_number_of_bars(self):
        """Verify the number of bars plotted for different `n_pairs` values."""
        random.seed(1)
        for i in [5, 10, 20]:
            ax = task_func(i)
            self.assertEqual(
                len(ax.patches),
                i,
                f"Expected {i} bars, but got {len(ax.patches)} bars.",
            )
    def test_labels_and_title(self):
        """Verify the labels and the title of the plotted bar chart."""
        random.seed(2)
        _ = task_func(15)
        fig = plt.gcf()
        axes = fig.gca()
        self.assertEqual(
            axes.get_xlabel(), "Letter:Number Pairs", "X label is incorrect."
        )
        self.assertEqual(axes.get_ylabel(), "Counts", "Y label is incorrect.")
        self.assertEqual(
            axes.get_title(), "Random Letter:Number Pairs Chart", "Title is incorrect."
        )
    def test_invalid_n_pairs(self):
        """Test the function with invalid `n_pairs` values."""
        random.seed(3)
        with self.assertRaises(ValueError):
            task_func(27)
        with self.assertRaises(ValueError):
            task_func(0)
    def test_valid_pairs(self):
        """Verify that the pairs generated are valid and correspond to the expected letter:number format."""
        random.seed(4)
        ax = task_func(5)
        expected_pairs = ["a:1", "b:2", "c:3", "d:4", "e:5"]
        generated_pairs = [bar.get_label() for bar in ax]
        for expected_pair in expected_pairs:
            self.assertIn(
                expected_pair,
                generated_pairs,
                f"Expected pair {expected_pair} not found in plotted pairs.",
            )
testcases = TestCases()
testcases.test_valid_pairs()
