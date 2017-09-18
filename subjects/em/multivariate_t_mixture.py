import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import cholesky, solve_triangular, LinAlgError, inv
from scipy.special import gammaln, digamma, logsumexp

class MultivariateTMixture:
    def __init__(self, n_components=1, max_iter=16, random_state=None, reg_covar=1e-6):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.reg_covar = reg_covar

    def _initialize_parameters(self, X):
        n_components = self.n_components
        n_samples, n_features = X.shape

        # initializing tau
        tau = np.ones((n_samples, n_components)) * .5
        # label = KMeans(n_clusters=n_components, random_state=self.random_state).fit(X).labels_
        # tau[np.arange(n_samples), label] = 1

        # initializng u
        u = np.ones((n_samples, self.n_components))

        return tau, u

    def _compute_precision_cholesky(self, sigma):
        n_components, n_features, _ = sigma.shape

        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(sigma):
            try:
                cov_chol = cholesky(covariance, lower=True)
            except LinAlgError:
                raise ValueError('covariance: {0}'.format(covariance))
            precisions_chol[k] = solve_triangular(cov_chol, np.eye(n_features), lower=True).T
        return precisions_chol

    def _estimate_covariances(self, X, mu, pi, tau, u):
        n_components, n_features = mu.shape
        sigma = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - mu[k]
            sigma[k] = np.dot(tau[:, k] * u[:, k] * diff.T, diff) / pi[k]
            sigma[:: n_features + 1] += self.reg_covar

        return sigma

    def _estimate_parameters(self, X, tau, u):
        """

        Returns
        -------
        pi :

        means :

        covariances :
        """
        pi = tau.sum(0)  # pi without normalizing
        mu = np.dot(tau.T * u.T, X) / np.sum(tau * u, 0)[:, np.newaxis]
        sigma = self._estimate_covariances(X, mu, pi, tau, u)

        return pi, mu, sigma

    #     def _estimate_dof(self, tau, u):
    #         return -digamma(self.df * .5) + log(self.df *.5) \
    #                 + (tau * (log(u) - u)).sum()/tau.sum() + 1 + (digamma((v+self.p)/2.)-log((v+self.p)/2.))

    def _compute_q1(self, tau):
        return tau * np.log(self.pi)

    def _compute_q2(self, df, tau, u):
        q2 = np.empty(tau.shape)
        for k in range(self.n_components):
            q2[:, k] = tau[:, k] * self._compute_q2_helper(df[k], u[:, k])
        return q2

    def _compute_q2_helper(self, df, u):
        half_df = df * .5
        half_feat = self.n_features * .5

        return -gammaln(half_df) + half_df * np.log(half_df) \
               + half_df * (digamma(half_df + half_feat) - np.log(half_df + half_feat) \
                            + np.sum(np.log(u) - u))

    def _compute_q3(self, tau, u, mahalanobis):
        log_det_chol = (np.sum(np.log(
            self.precisions_chol.reshape(
                self.n_components, -1)[:, ::self.n_features + 1]), 1))

        log_prob = mahalanobis * u
        q3 = .5 * (self.n_features * (np.log(u) - np.log(2 * np.pi)) - log_prob) + log_det_chol
        return q3 * tau

    def _compute_mahalanobis(self, X):
        n_samples, _ = X.shape

        mahalanobis = np.empty((n_samples, self.n_components))
        for k, (mu, prec_chol) in enumerate(zip(self.means, self.precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            mahalanobis[:, k] = np.sum(np.square(y), axis=1)
        return mahalanobis

    def _compute_u(self, mahalanobis):
        return (self.df + self.n_features) / (mahalanobis + self.df)

    def fit(self, X):
        n_samples, n_features = X.shape
        self.n_features = n_features

        tau, u = self._initialize_parameters(X)

        pi, means, sigmas = self._estimate_parameters(X, tau, u)
        pi /= n_samples  # normalizing
        self.pi = pi

        df = np.ones(self.n_components) * 4

        self.means = means
        self.sigmas = sigmas
        self.df = df

        # precision cholesky
        self.precisions_chol = self._compute_precision_cholesky(sigmas)

        likelihood = []
        for _ in range(self.max_iter):
            log_prob_norm, tau, u = self._e_step_1(X, tau, u)
            self._cm_step_1(X, tau, u)

            #             print '1#'
            #             print tau
            #             print '\nu'
            #             print u

            print '\nmean'
            print self.means

            u = self._e_step_2(X)
            #             print '\nu'
            #             print u
            self._cm_step_2(np.exp(tau), u)

            likelihood.append(np.mean(log_prob_norm))
            tau = np.exp(tau)
        self.likelihoodd = likelihood

    def _e_step_1(self, X, tau, u):
        """ E-step 01 - It calculates the loglikelihood under the current parameters
            and estimates the parameters tau and u.

        Parameters
        ----------
        X : array-like, shape(n_components, n_features)

        Returns
        -------
        tau : array-like, shape(n_samples, n_components)

        u   : array-like, shape(n_samples, n_components)

        """
        n_samples, n_features = X.shape

        q1 = self._compute_q1(tau)  # precisa do pi aqui
        q2 = self._compute_q2(self.df, tau, u)  # precisa das features e do DF

        mahalanobis = self._compute_mahalanobis(X)
        q3 = self._compute_q3(tau, u, mahalanobis)

        weighted_log_prob = q1 + q2 + q3
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)

        # estimating 'u'
        new_u = self._compute_u(mahalanobis)

        # returning tau, u
        return log_prob_norm, weighted_log_prob - log_prob_norm[:, np.newaxis], new_u

    def _cm_step_1(self, X, tau, u):
        n_samples, _ = X.shape

        pi, self.means, self.sigmas = self._estimate_parameters(X, np.exp(tau), u)
        pi /= n_samples  # normalizing
        self.pi = pi

        self.precisions_chol = self._compute_precision_cholesky(self.sigmas)

    def _e_step_2(self, X):
        # computing the mahalanobis distance
        # with the parameters from cm_step_1
        mahalanobis = self._compute_mahalanobis(X)
        return self._compute_u(mahalanobis)

    def _cm_step_2(self, tau, u):
        n = tau.sum(axis=0)
        tau_u = np.sum(tau * (np.log(u) - u), axis=0)

        new_df = np.empty(self.df.shape)
        for k in range(self.n_components):
            best = (self.df[k], 0)
            step = 10.
            for __ in range(5):
                for _ in np.arange(best[0], best[0] + step, step / 10):
                    result = -digamma(_ * .5) + np.log(_ * .5) + 1 + digamma(((_ + 2) * .5)) - np.log(((_ + 2) * .5)) + tau_u[k] / n[k]
                    if result > 0:
                        best = (_, result)
                    else:
                        break
                step /= 10.
            new_df[k] = best[0]
        self.df = new_df