# Source
## https://github.com/statsmodels/statsmodels/blob/master/statsmodels/sandbox/distributions/multivariate.py#L90
import numpy as np
from numpy import log

from scipy.special import gamma, digamma
from scipy.linalg import inv, det

from sklearn.cluster import KMeans

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
        for _ in x-mu:
            yield _
    
    # probability density function
    def pdf(self, x):
        "Probability of a data point given the current parameters"
        if type(x) != np.ndarray:
            x = np.array(x)
        if not hasattr(self, 'p') or self.p != x.shape[0]:
            self.p = x.shape[1]
            self.gamma1 = gamma(self.df/2.) ** -1
            self.gamma2 = gamma((self.df+self.p)/2.)
            self.const = 1./np.sqrt((2.*np.pi)**self.p * det(self.sigma))
        
        delta = []
        inv_sigma = inv(self.sigma)
        for _ in self.__delta(x, self.mu):
            delta.append(_.dot(inv_sigma).dot(_))
        delta = np.array(delta)
        
        return self.gamma1 * self.gamma2 * self.const * (1+delta/self.df) ** (-(self.df+self.p)/2.)
    
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
        
        clf = KMeans(n_clusters=self.p)
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
            
    def __calc_u2(self, X):
        # E-Step: Calculating u
        self.u = {}
        for idx in range(self.n_components):
            _ = []
            for delta in (X-self.mixes[idx].mu):
                _.append(delta.dot(inv(self.mixes[idx].sigma)).dot(delta))
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
            self.mixes[idx].mu, self.mixes[idx].sigma = self.__estimate_parameters(X, self.u[idx], self.weights[idx])

    def __delta(self, X, mu):
        for idx, _ in enumerate(X - mu):
            yield idx, _.reshape(-1, 1)

    def __estimate_parameters(self, X, u, tau):
        tau_u = tau * u
        mu_ = (tau_u * X).sum(axis=0) / tau_u.sum() # Here I could improve somehow...
        
        cov_ = np.array([[0,0], [0,0]], dtype=np.float32)
        for idx, delta in self.__delta(X, mu_):
            #delta = delta.reshape(-1, 1)
            cov_ += (tau[idx] * u[idx] * delta).dot(delta.T)
        
        cov_ /= tau.sum()
        
        return mu_, cov_

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
        result = []
        for idx, mix in enumerate(self.mixes):
            result.append(mix.pdf(X) * self.pi)

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