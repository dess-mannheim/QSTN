import pandas as pd
from typing import Dict, Any
import re


def create_one_dataframe(parsed_results: Dict[Any, pd.DataFrame]) -> pd.DataFrame:
    """Concatenates a dictionary of DataFrames into a single DataFrame.

    Args:
        parsed_results (Dict[Any, pd.DataFrame]): A dictionary mapping objects
            to DataFrames. Each key must be an object that has an
            `interview_name` attribute (e.g., a custom class instance). The
            values are the pandas DataFrames to be merged.

    Returns:
        pd.DataFrame: A single DataFrame containing the vertically concatenated
        data from all input DataFrames. Returns an empty DataFrame if the
        input dictionary is empty.
    """
    dataframes_to_concat = []

    for key, df in parsed_results.items():
        temp_df = df.copy()

        temp_df.insert(0, "interview_name", key.interview_name)

        dataframes_to_concat.append(temp_df)
    if not dataframes_to_concat:
        return pd.DataFrame()

    return pd.concat(dataframes_to_concat, ignore_index=True)


def safe_format_with_regex(template_string: str, data: dict) -> str:
    """
    Safely substitutes {{variable}} style placeholders using a regex.
    """

    def replacer(match):
        result = data.get(match.group(0), match.group(0))
        return result

    pattern = re.compile(r"\{\{(.*?)\}\}")

    return pattern.sub(replacer, template_string)
