import pandas as pd
from src.application.training.pipeline import run

def run_pipeline_with_df(df: pd.DataFrame):
    return run(df)
