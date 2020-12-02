from finance_ml.labeling import get_barrier_labels, cusum_filter
from finance_ml.stats import get_vol

vol = get_vol(close)
trgt = vol
timestamps = cusum_filter(close, vol)
labels = get_barrier_labels(close, timestamps, trgt, sltp=[1, 1],
                            num_days=1, min_ret=0, num_threads=16)
print(labels.show())