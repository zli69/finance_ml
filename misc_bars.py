##https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
##https://towardsdatascience.com/information-driven-bars-for-financial-machine-learning-imbalance-bars-dda9233058f0#:~:text=Imbalance%20bars%20are%20generated%20by,the%20beginning%20of%20each%20bar).
'''
We must aim for a bar representation in which each bar contains the same amount of information,
however time-based bars will oversample slow periods and undersample high activity periods.
To avoid this problem, the idea is to sample observations as a function of market activity.
'''

#Setup....

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
# raw trade data from https://public.bitmex.com/?prefix=data/trade/
data = pd.read_csv('data/20181127.csv')
data = data.append(pd.read_csv('data/20181128.csv')) # add a few more days
data = data.append(pd.read_csv('data/20181129.csv'))
data = data[data.symbol == 'XBTUSD']
# timestamp parsing
data['timestamp'] = data.timestamp.map(lambda t: datetime.strptime(t[:-3], "%Y-%m-%dD%H:%M:%S.%f"))

#TimeBars
def compute_vwap(df):
    q = df['foreignNotional']
    p = df['price']
    vwap = np.sum(p * q) / np.sum(q)
    df['vwap'] = vwap
    return df
data_timeidx = data.set_index('timestamp')
data_time_grp = data_timeidx.groupby(pd.Grouper(freq='15Min'))
num_time_bars = len(data_time_grp) # comes in handy later
data_time_vwap = data_time_grp.apply(compute_vwap)

#Tick Bars
total_ticks = len(data)
num_ticks_per_bar = total_ticks / num_time_bars
num_ticks_per_bar = round(num_ticks_per_bar, -3) # round to the nearest thousand
data_tick_grp = data.reset_index().assign(grpId=lambda row: row.index // num_ticks_per_bar)
data_tick_vwap =  data_tick_grp.groupby('grpId').apply(compute_vwap)
data_tick_vwap.set_index('timestamp', inplace=True)

#Volume Bars
data_cm_vol = data.assign(cmVol=data['homeNotional'].cumsum())
total_vol = data_cm_vol.cmVol.values[-1]
vol_per_bar = total_vol / num_time_bars
vol_per_bar = round(vol_per_bar, -2) # round to the nearest hundred
data_vol_grp = data_cm_vol.assign(grpId=lambda row: row.cmVol // vol_per_bar)
data_vol_vwap =  data_vol_grp.groupby('grpId').apply(compute_vwap)
data_vol_vwap.set_index('timestamp', inplace=True)

#Dollar Bars
# code omitted for brevity
# same as volume bars, except using data['foreignNotional'] instead of data['homeNotional']

#Imbalance Bars

def convert_tick_direction(tick_direction):
    if tick_direction in ('PlusTick', 'ZeroPlusTick'):
        return 1
    elif tick_direction in ('MinusTick', 'ZeroMinusTick'):
        return -1
    else:
        raise ValueError('converting invalid input: '+ str(tick_direction))
data_timeidx['tickDirection'] = data_timeidx.tickDirection.map(convert_tick_direction)
#Compute signed flows at each tick:
data_signed_flow = data_timeidx.assign(bv = data_timeidx.tickDirection * data_timeidx.size)

#from fast_ewma import _ewma
from pandas import fast_ewma as _ewma

abs_Ebv_init = np.abs(data_signed_flow['bv'].mean())
E_T_init = 500000  # 500000 ticks to warm up


def compute_Ts(bvs, E_T_init, abs_Ebv_init):
    Ts, i_s = [], []
    i_prev, E_T, abs_Ebv = 0, E_T_init, abs_Ebv_init

    n = bvs.shape[0]
    bvs_val = bvs.values.astype(np.float64)
    abs_thetas, thresholds = np.zeros(n), np.zeros(n)
    abs_thetas[0], cur_theta = np.abs(bvs_val[0]), bvs_val[0]
    for i in range(1, n):
        cur_theta += bvs_val[i]
        abs_theta = np.abs(cur_theta)
        abs_thetas[i] = abs_theta

        threshold = E_T * abs_Ebv
        thresholds[i] = threshold

        if abs_theta >= threshold:
            cur_theta = 0
            Ts.append(np.float64(i - i_prev))
            i_s.append(i)
            i_prev = i
            E_T = _ewma(np.array(Ts), window=np.int64(len(Ts)))[-1]
            abs_Ebv = np.abs(_ewma(bvs_val[:i], window=np.int64(E_T_init * 3))[-1])  # window of 3 bars
    return Ts, abs_thetas, thresholds, i_s


Ts, abs_thetas, thresholds, i_s = compute_Ts(data_signed_flow.bv, E_T_init, abs_Ebv_init)

#Aggregate the ticks into groups based on computed boundaries
n = data_signed_flow.shape[0]
i_iter = iter(i_s + [n])
i_cur = i_iter.__next__()
grpId = np.zeros(n)
for i in range(1, n):
    if i <= i_cur:
        grpId[i] = grpId[i-1]
    else:
        grpId[i] = grpId[i-1] + 1
        i_cur = i_iter.__next__()

#Put it altogether
data_dollar_imb_grp = data_signed_flow.assign(grpId = grpId)
data_dollar_imb_vwap = data_dollar_imb_grp.groupby('grpId').apply(compute_vwap).vwap