from multivariate_t_mixture import *
import matplotlib.pyplot as plt

def multivariate_t_rvs(m, S, df=np.inf, n=1):
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]

np.random.seed(0)

mu  = [0, 0]
cov = np.eye(2)
X1   = multivariate_t_rvs(mu, cov, 14, 100)

mu  = [1., 1.]
cov = [[.4, 0], [0, .4]]
X2 = multivariate_t_rvs(mu, cov, 29, 100)

X = np.concatenate([X1, X2])

t = MultivariateTMixture(2, 16)
t.fit(X)

print 'df'
print t.df

print '\nsigma'
print t.sigmas

print '\nmu'
print t.means