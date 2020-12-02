import pandas as pd
import numpy as np


def apply_func(x):
    return x ** 2


def func(df, timestamps, f):
    df_ = df.loc[timestamps]
    for idx, x in df_.items():
        df_.loc[idx] = f(x)
    return df_


df = pd.Series(np.random.randn(10000))
from finance_ml.multiprocessing import mp_pandas_obj

results = mp_pandas_obj(func, pd_obj=('timestamps', df.index),
                        num_threads=24, df=df, f=apply_func)
print(results.head())