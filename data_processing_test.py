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

"""Tests for data_processing module."""

import io
import unittest

import data_processing
import pandas as pd


class TestDataProcessing(unittest.TestCase):

  def test_load_dataframe(self):
    csv_data_string = "col1,col2\nval1,val2\nval3,val4"
    csv_file_like_object = io.StringIO(csv_data_string)
    df = data_processing.load_dataframe(csv_file_like_object)
    self.assertIsInstance(df, pd.DataFrame)
    self.assertEqual(len(df), 2)
    self.assertListEqual(list(df.columns), ["col1", "col2"])
    self.assertEqual(df.iloc[0]["col1"], "val1")

  def test_preprocess_dataframe_remove_html_true(self):
    data = {"col_with_html": ["<p>text1</p>", "<b>text2</b>"]}
    df = pd.DataFrame(data)
    processed_df = data_processing.preprocess_dataframe(df, remove_html=True)
    self.assertEqual(processed_df.iloc[0]["col_with_html"], "text1")
    self.assertEqual(processed_df.iloc[1]["col_with_html"], "text2")

  def test_preprocess_dataframe_remove_html_false(self):
    data = {"col_with_html": ["<p>text1</p>", "<b>text2</b>"]}
    df = pd.DataFrame(data)
    processed_df = data_processing.preprocess_dataframe(df, remove_html=False)
    self.assertEqual(processed_df.iloc[0]["col_with_html"], "<p>text1</p>")
    self.assertEqual(processed_df.iloc[1]["col_with_html"], "<b>text2</b>")

  def test_convert_df(self):
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    csv_bytes = data_processing.convert_df(df)
    expected_csv_string = "col1,col2\n1,a\n2,b\n"
    self.assertEqual(csv_bytes.decode("utf-8"), expected_csv_string)

  def test_convert_df_keeps_utf_8_encoding(self):
    df = pd.DataFrame({"col1": ["ü", "a"]})
    csv_bytes = data_processing.convert_df(df)
    expected_csv_string = "col1\nü\na\n"
    self.assertEqual(csv_bytes.decode("utf-8"), expected_csv_string)


if __name__ == "__main__":
  unittest.main()
