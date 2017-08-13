# Source
## https://github.com/statsmodels/statsmodels/blob/master/statsmodels/sandbox/distributions/multivariate.py#L90
import numpy as np
from scipy.special import gamma
from scipy.linalg import inv, det

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
        for _ in x-self.mu:
            delta.append(_.dot(inv(self.sigma)).dot(_))
        delta = np.array(delta)

        return self.gamma1 * self.gamma2 * self.const * (1+delta/self.df) ** (-(self.df+self.p)/2.)
    
    def __repr__(self):
        return 'mu: %s;\n df: %s;\n sigma: %s' % (self.mu, self.df, self.sigma)