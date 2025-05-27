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

"""Functions to render UI elements."""
from typing import Any, Optional, Dict

import llm_services
import pandas as pd
import streamlit as st


def render_sidebar():
  """Renders the sidebar with configuration options."""
  with st.sidebar:
    st.subheader("Configuration")
    gcp_id = st.text_input(
        "Google Cloud Project ID", placeholder="your-gcp-project-id"
    )
    region = st.text_input(
        "Region"
        " ([Options](https://cloud.google.com/vertex-ai/docs/general/locations))",
        value="us-central1",
        help=(
            "Not all regions support all models. Please check the provided link"
            " and select the model accordingly."
        ),
    )
    llm_model_name = (
        st.selectbox("Gemini model", llm_services.AVAILABLE_MODELS) or ""
    )
    temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.1,
        format="%.1f",
    )
  return gcp_id, region, llm_model_name, temperature


def render_dataframe(
    dataframe: pd.DataFrame, column_config: Optional[Dict[str, Any]] = None
) -> None:
  """Renders the given DataFrame in a streamlit container."""
  with st.container():
    st.dataframe(dataframe, column_config=column_config, hide_index=True)


def selectable_dataframe(
    dataframe: pd.DataFrame, column_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
  """(Re-)renders a dataframe and returns the selected rows.

  This function provides a workaround for missing streamlit functionality to
  get user selected rows from a DataFrame / Data editior. It (re-)renders a
  given DataFrame on the UI with all the selected rows and returns those
  selected rows as a new DataFrame.

  Args:
      dataframe (pd.DataFrame): Input DataFrame.
      column_config (dict): Optional column configuration for the Data editor.

  Returns:
      pd.DataFrame: Selected rows from the Data editor.
  """
  selections_df = dataframe.copy()
  selections_df.insert(0, "Select", False)

  results_view = st.data_editor(
      selections_df,
      use_container_width=True,
      hide_index=True,
      column_config=column_config,
  )

  selected_rows = results_view[results_view.Select]
  return selected_rows.drop("Select", axis=1)


def display_progressive_results(current_results_list, results_placeholder):
  if not current_results_list:
    results_placeholder.empty()
    return

  df = pd.DataFrame.from_records(data=current_results_list)
  results_placeholder.dataframe(df, hide_index=True, use_container_width=True)
