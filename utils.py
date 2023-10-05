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
"""Utility functions for the app."""
import re
from typing import Any, Iterable

import pandas as pd


def replace_separators(
    value: Any, separators: Iterable, new_separator: str = ", "
) -> Any:
    """Replaces all given separators with a new separator.

    Utility function that can replace any given separator in a string with
        another. e.g. a|b -> a, b.

    Args:
        value (Any): Input to replace separators in.
        separators (Iterable): A collection of separators that should be
            replaced.
        new_separator (str, optional): The separator to replace with. Defaults
        to ", ".

    Returns:
        Any: The new value.
    """
    if isinstance(value, str):
        for separator in separators:
            value = value.replace(separator, new_separator)
    return value


def remove_html_tags(value: Any) -> Any:
    """Removes all html tags in a string.

    Args:
        value (Any): Input string to replace in.

    Returns:
        Any: String without html characters.
    """
    if isinstance(value, str):
        return re.sub("<.*?>", "", value)
    return value


def row_to_custom_str(row: Any, ignore_empty: bool) -> str:
    """Converts a DataFrame row to a comma-separated string.

    For a given row of a tabular dataset, this function returns a string
    representation of it. It is also possible to ignore empty column values
    within the tabular data using the ignore_empty boolean.

    Args:
        row (Any): Input DataFrame row
        ignore_empty (bool): How to treat empty columns in the string
            representation.

    Returns:
        str: A string representation of tabular data.
    """
    custom_str = ""
    if ignore_empty:
        ignore = ["nan", "NAN", "NaN"]
        custom_str = [
            f"{col}: {value}"
            for col, value in row.items()
            if pd.notna(value) and value not in ignore
        ]
    else:
        custom_str = [f"{col}: {value}" for col, value in row.items()]
    return ", ".join(custom_str)
