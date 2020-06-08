from __future__ import division
import numpy as np
from scipy import stats
import math

def online_changepoint_detection(data, hazard_func, observation_likelihood):
    maxes = np.zeros(len(data) + 1)
    
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    
    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = observation_likelihood.pdf(x)
        
        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t+1)))
       
        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)
        
        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)
        
        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        
        # Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)
    
        maxes[t] = R[:, t].argmax()
    return R, maxes


def constant_hazard(lam, r):
    return 1/lam * np.ones(r.shape)


def gaussian_hazard(mu, sigma, r):
    # mu and sigma are the hyperparameters of the cecum event happening. In our case 12000 frames correspond to 6.5min where cecum is typically observed
    # Hazard function = pdf/survival = pdf/(1-cdf)
    # Greater run length will have greater hazard. So we use stats.norm.pdf(r) directly which will return an array of hazard values for each run length
    pdf = stats.norm.pdf(r,mu,sigma) # Computes probability for each runlength starting from 0 to t
    survival = 1-stats.norm.cdf(r,mu,sigma) # Computes survival for each runlength starting from 0 to t
    hazard_arr =  pdf / survival # Computes hazard value for each runlength starting from 0 to t
    #hazard_arr = hazard*np.ones(r.shape) # Apply the same hazard for all vaules of column t in R[] ---> Not reqired since the previous line itself is an array of r.shape values
    return hazard_arr

class StudentT:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):
        return stats.t.pdf(x=data, 
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha *
                               self.kappa)))

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))
            
        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0

class NormalKnownPrecision:
    # Normal --> use this when variance is known and mean is unknown / changing for the incoming data
    def __init__(self, mu, prec):
        self.mu0 = self.mu = np.array([mu])
        self.prec0 = self.prec = np.array([prec])

    def pdf(self, data):
        return stats.norm.pdf(data,self.mu, 1/self.prec + 1)

    def update_theta(self, data, t):
        offsets = np.arange(1, t+2) # To calculate new mean params at each node of the next time step,  we need to multiply each of the previous mean value in self.mu array (2u1+data/3, 3u2+data/4 etc.,), we have offsets
        muT0 = np.concatenate((self.mu0, (offsets*self.mu + data)/(offsets+1)))
        precT0 = np.concatenate((self.prec0, self.prec+self.prec0))
        #precT0 = np.concatenate((self.prec0,  math.sqrt(t / ( (t-1)/self.prec**2 + (data-newMu)*(data-self.mu))) ))

        self.mu = muT0
        self.prec = precT0


class Poisson:
    def __init__(self, k, theta):
        self.k0 = self.k = np.array([k])
        self.theta0 = self.theta = np.array([theta])

    def pdf(self, data):
        return stats.nbinom.pmf(data,self.k, 1/(1+self.theta))

    def update_theta(self, data):
        kT0 = np.concatenate((self.k0, self.k+data))
        thetaT0 = np.concatenate((self.theta0, self.theta/(1+self.theta)))

        self.k = kT0
        self.theta = thetaT0
