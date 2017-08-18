# Source
## https://github.com/statsmodels/statsmodels/blob/master/statsmodels/sandbox/distributions/multivariate.py#L90
import numpy as np
from numpy import log

from scipy.special import gamma, digamma
from scipy.linalg import inv, det

from sklearn.cluster import KMeans
import scipy.linalg.blas as FB

def get_random(X):
    """Get a random sample from X.
    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
    
    Returns
    -------
    array-like, shape (1, n_features)
    """
    size = len(X)
    idx = np.random.choice(range(size))
    return X[idx]

#written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

# This one was written by me
class multivariate_t:
    """Calculate the multivariate t-Distribution probability
    Parameters
    ----------
    mu: array_like; 1 by n
        mean of the distribution
    sigma: array_like; n by n
        covariance matrix
    df: scalar
        degrees of freedom
    """

    def __init__(self, mu, sigma, df):
        self.mu = mu
        self.sigma = sigma
        self.df = df

    def __delta(self, x, mu):
        for idx, _ in enumerate(x-mu):
            yield idx, _.reshape(-1, 1)

    # probability density function
    def pdf(self, x):
        "Probability of a data point given the current parameters"
        if type(x) != np.ndarray:
            x = np.array(x)
        if not hasattr(self, 'p') or self.p != x.shape[0]:
            self.p = x.shape[1]
            self.const = (self.df*np.pi)**(self.p/2.)

        delta = x - self.mu
        delta = delta.dot(inv(self.sigma))

        result = []
        for idx, _ in self.__delta(x, self.mu):
            result.append(delta[idx].reshape(1, -1).dot(_)[0])
        delta = np.array(result)
        
        top = gamma((self.df+self.p)/2.) * np.sqrt(det(self.sigma))**-1
        bottom = gamma(self.df/2.) * self.const
        
        bottom *= (1+delta/self.df) ** ((self.df+self.p)/2.)
        
        return top/bottom
    
    def __repr__(self):
        return 'mu: %s;\n df: %s;\n sigma: %s' % (self.mu, self.df, self.sigma)

class MultivariateTMixture:
    """
    Parameters
    ----------
    n_components: number of components
    """
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        self.p = X.shape[1]
        
        clf = KMeans(n_clusters=self.n_components)
        clf.fit(X)
        
        xmin = get_random(X)
        xmax = get_random(X)
        xcov = np.cov(X.T.copy())
        self.mixes = [multivariate_t(mu=clf.cluster_centers_[_], sigma=xcov, df=4) for _ in range(self.n_components)]
        self.pi    = [1./self.n_components for _ in range(self.n_components)]
    
    def __calc_tau(self, X):
        # E-step: Calculating tau
        self.weights = {}
        for idx in range(self.n_components):
            self.weights[idx] = (self.mixes[idx].pdf(X) * self.pi[idx]).reshape(-1, 1)
        weights_total = sum(self.weights.values())
        self.weights_total = weights_total
        
        # normalizing weights
        for idx, _ in enumerate(self.weights):
            self.weights[idx] /= weights_total
    
    def __calc_u(self, X):
        # E-Step: Calculating u
        self.u = {}
        for idx in range(self.n_components):
            _ = []
            inv_sigma = inv(self.mixes[idx].sigma)
            for delta in (X-self.mixes[idx].mu):
                _.append(delta.dot(inv_sigma).dot(delta))
            _ = np.array(_)
            self.u[idx] = (self.mixes[idx].df + self.p)/(self.mixes[idx].df + _)
            self.u[idx] = self.u[idx].reshape(-1, 1)

    def e_step1(self, X):
        self.__calc_tau(X)
        self.__calc_u(X)
    
    def e_step2(self, X):
        self.__calc_u(X)

    def cm_step1(self, X):
        for idx, _ in enumerate(self.mixes):
            # Assigning mu and cov
            self.mixes[idx].mu    = self.__mu(X, self.u[idx], self.weights[idx])
            self.mixes[idx].sigma = self.__cov(X, self.u[idx], self.weights[idx], self.mixes[idx].mu)

    def __delta(self, X, mu):
        for idx, _ in enumerate(X - mu):
            yield idx, _.reshape(-1, 1)

    def __mu(self, X, u, tau):
        tau_u = tau * u
        return (tau_u * X).sum(axis=0) / tau_u.sum()  # Here I could improve somehow...

    def __cov(self, X, u, tau, mu_):
        cov_ = []
        for idx, delta in self.__delta(X, mu_):
            result = FB.dgemm(alpha=1.0, a=(tau[idx] * u[idx] * delta), b=delta, trans_b=True)
            cov_.append(result)
        cov_ = np.array(cov_)

        return cov_.sum(axis=0) / tau.sum()

    def __estimate_dof(self, v, u, tau):
        return -digamma(v/2.) + log(v/2.) + (tau * (log(u) - u)).sum()/tau.sum() + 1 + (digamma((v+self.p)/2.)-log((v+self.p)/2.))
    
    def cm_step2(self, X):
        for idx, _ in enumerate(self.mixes):
            arange = np.arange(_.df, _.df+3, .01)
            for df in arange:
                solution = self.__estimate_dof(df, self.u[idx], self.weights[idx])
                if solution < 0+1e-4 and solution > 0-1e-4:
                    # Assigning degrees of freedom
                    _.df = df
                    break

        for idx, pi in enumerate(self.pi):
            self.pi[idx] = sum(self.weights[idx])/len(self.weights[idx])
    
    def pdf(self, X):
        result = 0
        for idx, mix in enumerate(self.mixes):
            result += mix.pdf(X) * self.pi[idx]
        return result
    
    def log_likelihood(self, X):
        likelihood = 0
        for _ in self.pi:
            likelihood += len(X) * log(_)
        
        for mix in self.mixes:
            likelihood += log(mix.pdf(X)).sum()
        
        return likelihood
    
    def iterate(self, X):
        self.e_step1(X)
        self.cm_step1(X)
        self.e_step2(X)
        self.cm_step2(X)