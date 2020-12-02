# GARCH(1,1) Model in Python
#   uses maximum likelihood method to estimate (omega,alpha,beta)
# (c) 2014 QuantAtRisk, by Pawel Lachowicz; tested with Python 3.5 only

#http://www.quantatrisk.com/2014/10/23/garch11-model-in-python/
#http://www.quantatrisk.com/2013/03/30/garchpq-model-and-exit-strategy-for-intraday-algorithmic-traders/
#https://www.probabilitycourse.com/chapter10/10_1_0_basic_concepts.php

import numpy as np

r = np.array([0.945532630498276,
              0.614772790142383,
              0.834417758890680,
              0.862344782601800,
              0.555858715401929,
              0.641058419842652,
              0.720118656981704,
              0.643948007732270,
              0.138790608092353,
              0.279264178231250,
              0.993836948076485,
              0.531967023876420,
              0.964455754192395,
              0.873171802181126,
              0.937828816793698])

from arch import arch_model

garch11 = arch_model(r, p=1, q=1, rescale=False)
res = garch11.fit(update_freq=10)
print(res.summary())