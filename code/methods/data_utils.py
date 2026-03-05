import numpy as np
import pandas as pd

def load_data(market: str | None = None,
              file_path: str | None = None) -> pd.DataFrame:
    """
    Load tick data and compute mid price.
    """

    if file_path is None:
        if market is None:
            raise ValueError("Either market or file_path must be provided.")
        
        file_path = f"../data/raw_data/DAT_ASCII_{market}_T_202602.csv"

    df = pd.read_csv(file_path, header=None)
    df.columns = ["datetime", "bid", "ask", "volume"]

    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S%f")

    df = df.set_index("datetime")
    
    df.index = df.index.tz_localize("US/Eastern").tz_convert("UTC")
    
    df["mid_price"] = (df["bid"] + df["ask"]) / 2

    return df


def pre_avg(df: pd.DataFrame,
            column: str = "mid_price", 
            tick: bool = True, 
            k: int = 5,
            time_interval_ms: int = 100, 
            time_col: str = "datetime") -> pd.DataFrame:
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
    
    df = df.dropna(subset =["pre_avg"])   
    return df



def compute_returns(df: pd.DataFrame, column: str = "pre_avg") -> pd.DataFrame:
    """
    Compute returns from chosen column.
    """
    df = df.copy()
    df["r"] = np.log(df[column]).diff()
    df = df.dropna(subset=["r"]).reset_index(drop=True)
    return df

def filter_day(df: pd.DataFrame, day: str) -> pd.DataFrame:

    day = pd.to_datetime(day).date()

    return df[df["datetime"].dt.date == day]



def summarize_data(df: pd.DataFrame, time_col: str = "datetime") -> None:
    """Print basic summary statistics for tick data."""

    print("----- DATA SUMMARY -----")

    print("\nObservations:")
    print(len(df))

    print("\nTime range:")
    print(df[time_col].min(), "→", df[time_col].max())

    print("\nMissing values:")
    print(df.isna().sum())

    if {"bid", "ask"}.issubset(df.columns):
        spread = df["ask"] - df["bid"]
        print("\nSpread statistics:")
        print(spread.describe())

    if "mid_price" in df.columns:
        print("\nMid price statistics:")
        print(df["mid_price"].describe())

    if time_col in df.columns:
        time_diff = df[time_col].diff()
        print("\nTime gap statistics:")
        print(time_diff.describe())

    print("\n------------------------")
    


    