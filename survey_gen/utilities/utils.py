from typing import List, Dict, Optional

import pandas as pd

def convert_wide_survey_dataframe(df: pd.DataFrame, json_structure:List[str], order:Optional[List[int]] = None) -> pd.DataFrame:
    long_df = pd.wide_to_long(df, 
                            stubnames=json_structure, 
                            i=["batch"], 
                            j='question_id2', 
                            sep='', 
                            suffix='\\d+').reset_index()

    long_df = long_df.drop(columns='question_id').rename(columns={'question_id2': 'question_id'})

    if order:
        long_df["question_id"] = long_df.question_id.apply(lambda x: order[x])

    return long_df.sort_values(["batch", "question_id"])