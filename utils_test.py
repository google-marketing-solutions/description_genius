# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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

    def test_row_to_custom_str_removes_nan_values(self):
        row = pd.Series({"col1": "a", "col2": "nan"})
        expected_output = "col1: a"
        self.assertEqual(utils.row_to_custom_str(row, True), expected_output)

    def test_row_to_custom_str_removes_empty_values(self):
        row = pd.Series({"col1": "a", "col2": ""})
        expected_output = "col1: a"
        self.assertEqual(utils.row_to_custom_str(row, True), expected_output)

    def test_row_to_custom_str_removes_na_values(self):
        row = pd.Series({"col1": "a", "col2": pd.NA})
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


if __name__ == "__main__":
    unittest.main()
