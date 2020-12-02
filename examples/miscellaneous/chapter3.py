import pandas as pd
#C:\Users\staraustin\PycharmProjects\finance_ml\examples\data\TSLA1minETH.csv
df = pd.read_csv("C:/Users/staraustin/PycharmProjects/finance_ml/examples/data/TSLA1minETH.csv")
df.index = pd.DatetimeIndex(df['Date'].values)
close = df["Close"]

print(df.head())

import numpy as np
import pandas as pd

def get_daily_vol(close, span=100):
    use_idx  = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    use_idx0 = close.index.searchsorted(close.index)
    use_idx0=use_idx0[use_idx > 0]
    use_idx = use_idx[use_idx > 0]

    # Get rid of duplications in index
    #use_idx = np.unique(use_idx) 
    #use_idx0=np.unique(use_idx0)

    idx = pd.Series(close.index[use_idx0], index=close.index[use_idx])
    #idx=idx.drop_duplicates
    #https://stackoverflow.com/questions/13035764/remove-rows-with-duplicate-indices-pandas-dataframe-and-timeseries
    idx=idx[~idx.index.duplicated(keep='first')]
    ret = close.loc[idx.values].values / close.loc[idx.index] - 1
    vol = ret.ewm(span=span).std()
    return vol

vol = get_daily_vol(df["Close"])
print(vol.head())

import numbers

#book page 38.
def cusum_filter(close, h):
    # asssum that E y_t = y_{t-1}
    t_events = []
    s_pos, s_neg = 0, 0
    ret = close.pct_change().dropna()
    diff = ret.diff().dropna()
    # time variant threshold
    if isinstance(h, numbers.Number):
        h = pd.Series(h, index=diff.index)
    h = h.reindex(diff.index, method='bfill')
    h = h.dropna()
    for t in h.index:
        s_pos = max(0, s_pos + diff.loc[t])
        s_neg = min(0, s_neg + diff.loc[t])
        if s_pos > h.loc[t]:
            s_pos = 0
            t_events.append(t)
        elif s_neg < -h.loc[t]:
            s_neg = 0
            t_events.append(t)
    return pd.DatetimeIndex(t_events)

cusum_filter(df["Close"], 0.1)

vol = get_daily_vol(close)
sampled_idx = cusum_filter(close, vol)
print(sampled_idx)

print(df.shape)

def get_t1(close, t_events, num_days):
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
    return t1

t1 = get_t1(close, sampled_idx, num_days=1)
print(t1.shape)
print(t1.head())

def apply_ptslt1(close, events, ptsl, molecule):
    """Return datafram about if price touches the boundary"""
    # Sample a subset with specific indices
    _events = events.loc[molecule]
    # Time limit
    
    out = pd.DataFrame(index=_events.index)
    # Set Profit Taking and Stop Loss
    if ptsl[0] > 0:
        pt = ptsl[0] *  _events["trgt"]
    else:
        # Switch off profit taking
        pt = pd.Series(index=_events.index)
    if ptsl[1] > 0:
        sl = -ptsl[1] * _events["trgt"]
    else:
        # Switch off stop loss
        sl = pd.Series(index=_events.index)
    # Replace undifined value with the last time index
    time_limits = _events["t1"].fillna(close.index[-1])
    for loc, t1 in time_limits.iteritems():
        df = close[loc:t1]
        # Change the direction depending on the side
        df = (df / close[loc] - 1) * _events.at[loc, 'side']
        # print(df)
        # print(loc, t1, df[df < sl[loc]].index.min(), df[df > pt[loc]].index.min())
        out.at[loc, 'sl'] = df[df < sl[loc]].index.min()
        out.at[loc, 'pt'] = df[df > pt[loc]].index.min()
    out['t1'] = _events['t1'].copy(deep=True)
    return out


def get_3barriers(close, t_events, ptsl, trgt, min_ret=0, num_threads=1,
                  t1=False, side=None):
    # Get sampled target values
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]
    # Get time boundary t1
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)
    # Define the side
    if side is None:
        _side = pd.Series(1., index=trgt.index)
        _ptsl = [ptsl, ptsl]
    else:
        _side = side.loc[trgt.index]
        _ptsl = ptsl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': _side}, axis=1)
    events = events.dropna(subset=['trgt'])
    time_idx = apply_ptslt1(close, events, _ptsl, events.index)
    # Skip when all of barrier are not touched
    events['t1'] = time_idx.dropna(how='all').min(axis=1)
    events = events.drop('side', axis=1)
    return events

trgt = vol
events = get_3barriers(close, t_events=sampled_idx, trgt=trgt,
                       ptsl=1, t1=t1)
print(events.head())

def get_bins(events, close):
    # Prices algined with events
    events = events.dropna(subset=['t1'])
    px = events.index.union(events['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # Create out object
    out = pd.DataFrame(index=events.index)
    out['ret'] = px.loc[events['t1'].values].values / px.loc[events.index] - 1.
    if 'side' in events:
        out['ret'] *= events['side']
    out['bin'] = np.sign(out['ret'])
    if 'side' in events:
        out.loc[out['ret'] <= 0, 'bin'] = 0
    return out

bins = get_bins(events, close)
bins.head()

def drop_labels(events, min_pct=0.05):
    while True:
        df = events['bin'].value_counts(normalize=True)
        if df.min() > min_pct or df.shape[0] < 3:
            break
        print('dropped label', df.argmin(), df.min())
        events = events[events['bin'] != df.argmin()]
    return events

dropped_bins = drop_labels(bins)
print(bins.shape)
print(dropped_bins.shape)

bins = dropped_bins

def get_3barriers(close, t_events, ptsl, trgt, min_ret=0, num_threads=1,
                  t1=False, side=None):
    # Get sampled target values
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]
    # Get time boundary t1
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)
    # Define the side
    if side is None:
        _side = pd.Series(1., index=trgt.index)
        _ptsl = [ptsl, ptsl]
    else:
        _side = side.loc[trgt.index]
        _ptsl = ptsl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': _side}, axis=1)
    events = events.dropna(subset=['trgt'])
    time_idx = apply_ptslt1(close, events, _ptsl, events.index)
    # Skip when all of barrier are not touched
    time_idx = time_idx.dropna(how='all')
    events['t1_type'] = time_idx.idxmin(axis=1)
    events['t1'] = time_idx.min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    return events

def get_bins(events, close):
    # Prices algined with events
    events = events.dropna(subset=['t1'])
    px = events.index.union(events['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # Create out object
    out = pd.DataFrame(index=events.index)
    out['ret'] = px.loc[events['t1'].values].values / px.loc[events.index] - 1.
    if 'side' in events:
        out['ret'] *= events['side']
    out['bin'] = np.sign(out['ret'])
    # 0 when touching vertical line
    out['bin'].loc[events['t1_type'] == 't1'] = 0
    if 'side' in events:
        out.loc[out['ret'] <= 0, 'bin'] = 0
    return out

t1 = get_t1(close, sampled_idx, num_days=1)
events = get_3barriers(close, t_events=sampled_idx, trgt=trgt,
                       ptsl=1, t1=t1)
events.head()

print(events['t1_type'].unique())
print(events['t1_type'].describe())

bins = get_bins(events, close)
bins.head()

bins['bin'].value_counts()

import talib
import numpy as np


def macd_side(close):
    macd, signal, hist = talib.MACD(close.values)
    hist = pd.Series(hist).fillna(1).values
    return pd.Series(2 * ((hist > 0).astype(float) - 0.5), index=close.index[-len(hist):])

import numpy as np

vol = get_daily_vol(close)
sampled_idx = cusum_filter(close, vol)
t1 = get_t1(close, sampled_idx, num_days=1)
side =  macd_side(close)
events = get_3barriers(close, t_events=sampled_idx, trgt=vol,
                       ptsl=[1, 2], t1=t1, side=side)
events.head()

bins = get_bins(events, close)
bins.head()

bins['bin'].unique()

# In[93]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
x = np.hstack([events['side'].values[:, np.newaxis], close.loc[events.index].values[:, np.newaxis]])
y = bins['bin'].values
clf.fit(x, y)


# In[94]:


clf.predict(x)


# In[91]:


x.shape


# In[76]:


events['side'].values


# In[37]:


help(talib.MACD)


# In[42]:
macd, signal, hist = talib.MACD(close.values)
# In[45]:
np.max(macd[100:] - signal[100:]  - hist[100:] )


# In[49]:


macd[np.isfinite(macd)].shape


# In[51]:


signal = signal[np.isfinite(signal)]


# In[55]:


2 * ((signal > 0).astype(float) - 0.5)


# In[68]:


macd.fill(1)


# In[69]:


macd


# In[ ]:

