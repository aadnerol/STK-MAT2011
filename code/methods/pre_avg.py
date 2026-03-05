import numpy as np
import pandas as pd

def pre_avg(df: pd.DataFrame,
            column: str, 
            tick: bool = True, 
            k: int = 5,
            time_interval_ms: int = 100, 
            time_col: str = "datetime") -> pd.Dataframe:
    """Function for pre-averaging scheme.

    Args:
        df (pd.DataFrame): Tick data
        column (str): Name of column containing prices to be pre-averaged
        tick (bool, optional): Pre-average on tick. Defaults to True. If false, 
            pre-average on time
        k (int, optional): Size of pre-averaging block. Defaults to 5.
        time_interval_ms (int, optional): Length of pre-averaging time block. 
            Defaults to 100.
        time_col (str, optional): Name of column containing date/time. 
            Defaults to "datetime".

    Returns:
        pd.Dataframe: Dataframe with added "pre_avg" column
    """
    
    df = df.copy()
    
    if tick:
        df["pre_avg"] = df[column].rolling(window=k).mean()
    else:
        df[time_col] = pd.to_datetime(df[time_col]) 
        
        df = df.set_index(time_col)
        
        df["pre_avg"] = (
            df[column]
            .resample(f"{time_interval_ms}ms")
            .transform("mean")
        )

        df = df.reset_index()
        
    return df

