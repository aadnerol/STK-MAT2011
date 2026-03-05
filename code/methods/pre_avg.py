import numpy as np
import pandas as pd

def pre_avg(df: pd.DataFrame, 
            tick: bool = True, 
            k: int = 5,
            time_interval_ms: int = 100) -> np.ndarray:
    if tick == True:
        pass
    else: 
        pass 

#### Make function for pre-averaging: 
# P^bar_t = 1/k sum_{j=0}^{j=k}(P_{k-j})


# Se på dette
