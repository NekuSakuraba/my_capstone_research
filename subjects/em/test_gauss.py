import numpy as np
from sklearn.mixture import GaussianMixture

X = np.random.randn(6, 1)

clf = GaussianMixture(2)
clf.fit(X)