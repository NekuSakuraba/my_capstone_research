from multivariate_util import *

np.random.seed(0)

X = np.random.multivariate_normal([0,0], [[1,0], [0,1]], 200)
X = np.concatenate([X, np.random.multivariate_normal([.5, 1.], [[.5,0], [0,.5]], 200)])

t = MultivariateTMixture(2)
t.fit(X)

t.iterate(X)