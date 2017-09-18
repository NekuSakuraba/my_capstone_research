from multivariate_t_mixture import *
import matplotlib.pyplot as plt

np.random.seed(0)

X = np.random.multivariate_normal([0,0], [[1,0], [0,1]], 200)
X = np.concatenate([X, np.random.multivariate_normal([.5, 1.], [[.5,0], [0,.5]], 200)])

t = MultivariateTMixture(2, random_state=0, max_iter=10)
t.fit(X)

samples, features = X.shape
components = 2

print 'samples    {0}'.format(samples)
print 'features   {0}'.format(features)
print 'components {0}\n'.format(components)

print '\ndf {}'.format(t.df)

arr = np.arange(len(t.likelihoodd))
plt.scatter(arr, t.likelihoodd)
plt.show()