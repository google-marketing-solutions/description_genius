# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for utils."""
import unittest

import pandas as pd

import utils


class TestReplaceSeparators(unittest.TestCase):

  def test_replace_separators_replaces_pipes(self):
    pipe_separated_string = "a|b|c"
    comma_separated_string = utils.replace_separators(
        value=pipe_separated_string, separators=["|"]
    )
    self.assertEqual(comma_separated_string, "a, b, c")

  def test_replace_separators_replaces_pipes_and_dashes(self):
    pipe_separated_string = "a|b|c-d-e-f"
    comma_separated_string = utils.replace_separators(
        value=pipe_separated_string, separators=["|", "-"]
    )
    self.assertEqual(comma_separated_string, "a, b, c, d, e, f")


class TestRowToCustomStr(unittest.TestCase):

  def test_row_to_custom_str_returns_empty_string_for_empty_input(self):
    row = pd.Series()
    self.assertEqual(utils.row_to_custom_str(row, True), "")

  def test_row_to_custom_str_removes_none_values(self):
    row = pd.Series({"col1": "a", "col2": None})
    expected_output = "col1: a"
    self.assertEqual(utils.row_to_custom_str(row, True), expected_output)

  def test_row_to_custom_str_removes_actual_nan_values(
      self
  ):  # Renamed for clarity
    row = pd.Series({"col1": "a", "col2": pd.NA})  # Using pd.NA or np.nan
    expected_output = "col1: a"
    self.assertEqual(utils.row_to_custom_str(row, True), expected_output)

  def test_row_to_custom_str_keeps_literal_nan_string_if_not_empty_when_ignore_empty_true(
      self
  ):
    # This test's behavior changes with the suggested utils.py modification.
    # If "nan" should still be specifically ignored, utils.py needs adjustment.
    # Otherwise, "nan" as a string is valid and will be kept.
    row = pd.Series({"col1": "a", "col2": "nan"})
    expected_output_if_nan_is_valid_string = "col1: a, col2: nan"
    self.assertEqual(
        utils.row_to_custom_str(row, True),
        expected_output_if_nan_is_valid_string
    )

  def test_row_to_custom_str_removes_empty_values(self):
    row = pd.Series({"col1": "a", "col2": ""})
    expected_output = "col1: a"
    self.assertEqual(utils.row_to_custom_str(row, True), expected_output)

  def test_row_to_custom_str_removes_na_values(self):
    # This test is similar to test_row_to_custom_str_removes_actual_nan_values
    row = pd.Series({"col1": "a", "col2": pd.NA})
    expected_output = "col1: a"
    self.assertEqual(utils.row_to_custom_str(row, True), expected_output)

  def test_row_to_custom_str_removes_whitespace_only_values(self):
    row = pd.Series({"col1": "a", "col2": "   "})
    expected_output = "col1: a"
    self.assertEqual(utils.row_to_custom_str(row, True), expected_output)

  def test_row_to_custom_str_keeps_empty_value_when_ignore_empty_false(self):
    row = pd.Series({"col1": "a", "col2": "nan"})
    expected_output = "col1: a, col2: nan"
    self.assertEqual(utils.row_to_custom_str(row, False), expected_output)


class TestRemoveHtmlTags(unittest.TestCase):

  def test_remove_html_tags_removes_valid_html(self):
    html_str = "<b>hello</b>"
    self.assertEqual(utils.remove_html_tags(html_str), "hello")

  def test_remove_html_tags_does_not_remove_non_html_brackets(self):
    non_html_str = "<hello <<"
    self.assertEqual(utils.remove_html_tags(non_html_str), non_html_str)

  def test_remove_html_tags_handles_non_string_input(self):
    self.assertEqual(utils.remove_html_tags(123), 123)
    self.assertIsNone(utils.remove_html_tags(None), None)
    self.assertEqual(utils.remove_html_tags([1, 2]), [1, 2])


if __name__ == "__main__":
  unittest.main()
