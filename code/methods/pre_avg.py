import numpy as np
import pandas as pd

def pre_avg(df: pd.DataFrame,
            column: str, 
            tick: bool = True, 
            k: int = 5,
            time_interval_ms: int = 100, 
            time_col: str = "datetime") -> pd.Dataframe:
    
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

