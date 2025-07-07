import pandas as pd
from typing import Dict
from ..survey_manager import LLMSurvey

def extract_number_manual(key: str) -> int | None:
    i = len(key) - 1
    while i >= 0 and key[i].isdigit():
        i -= 1
    number_part = key[i + 1 :]
    return int(number_part) if number_part else None


def create_one_dataframe(parsed_results: Dict[LLMSurvey, pd.DataFrame]) -> pd.DataFrame:
    """
    Joins a dictionary of DataFrames into a single DataFrame.

    - The dictionary key's 'survey_name' attribute is added as the first column.
    - Handles different columns across DataFrames by creating a union of all columns.
    """
    dataframes_to_concat = []
    
    for key, df in parsed_results.items():
        temp_df = df.copy()
        
        temp_df.insert(0, 'survey_name', key.survey_name)
        
        dataframes_to_concat.append(temp_df)
    if not dataframes_to_concat:
        return pd.DataFrame()

    return pd.concat(dataframes_to_concat, ignore_index=True)