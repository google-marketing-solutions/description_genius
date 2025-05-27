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

"""Functions to process dataframes."""
import pandas as pd
import streamlit as st

import utils


def load_dataframe(csv_data) -> pd.DataFrame:
  """Loads a Pandas DataFrame from a given file object."""
  return pd.read_csv(csv_data)


def preprocess_dataframe(
    dataframe: pd.DataFrame, remove_html: bool = False
) -> pd.DataFrame:
  """Applies provided modifications to the DataFrame.

  Args:
    dataframe (pd.DataFrame): Input DataFrame.
    remove_html (bool, optional): If html should be removed.
      Defaults to False.

  Returns:
    pd.DataFrame: Processed DataFrame.
  """
  df_preprocessed = dataframe
  if remove_html:
    df_preprocessed = dataframe.map(utils.remove_html_tags)
  return df_preprocessed


@st.cache_data
def convert_df(dataframe: pd.DataFrame) -> bytes:
  """Creates a utf-8 encoded CSV from a given DataFrame."""
  return dataframe.to_csv(index=False).encode("utf-8")
