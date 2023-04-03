import numpy as np
from scipy.linalg import logm, expm
import random

class LIM:
    """
    A predicting model based on lead-lag linear regression

    Parameters
    ----------
    Init: np.ndarray
        An array containing initial condition
    
    data: np.ndarray
        An array containing data
        The lead-lag linear regression will be based on this array.

    lag: int
        The lag of lead-lag linear regression
        lag = 0 is not allowed.

    ntimestep: int
        The number of predicting timesteps
    """
    def __init__(self, Init: np.ndarray, data: np.ndarray, lag: int, ntimestep: int):
        self.Init = Init
        self.data = data
        self.lag  = lag
        self.nt   = ntimestep
        self.G    = None
        self.G1   = None
        self.e    = None
        self.out  = None
        
    def _calc_G1(self, G: np.ndarray, lag: int):
        """
        Calculating the lag 1 regression coefficient matrix by lag tau
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        return expm(logm(G)/lag)
    
    def _calc_G(self):
        """
        Calculating the regression coefficient matrix of lead time and lag time

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.lag != 0:
            x0 = np.copy(self.data[:, :-self.lag])
            xt = np.copy(self.data[:, self.lag:])
        else:
            x0 = np.copy(self.data[:])
            xt = np.copy(self.data[:])
        c0 = np.matmul(x0, x0.T)
        ct = np.matmul(xt, x0.T)
        G  = np.matmul(ct, np.linalg.inv(c0))
        self.G  = G
        self.G1 = self._calc_G1(self.G, self.lag)
        
    def _predict(self, PCs: np.ndarray, e: np.ndarray):
        """
        Using the coefficient matrix and white noise forcing to forecast

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        predicted = np.dot(self.G1, PCs) + e
        return predicted
    
    def _calc_e(self):
        """
        Calculating white noise forcing

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.e = np.zeros((self.data.shape[0], self.data.shape[1]-self.lag))
        for t_ in range(self.data.shape[1]-self.lag):
            predicted_ = self._predict(self.data[:, t_], 0)
            self.e[:, t_] = self.data[:, t_] - predicted_
            
    def build(self):
        """
        Building the model
        Call _calc_G() and _calc_e()

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._calc_G()
        self._calc_e()
    
    def run(self, stochastic = True):
        """
        Running the model and giving the predicted results

        Parameters
        ----------
        stochastic: bool
            If True, stochastic forcing will be added while integrating

        Returns
        -------
        None
        """
        self.out       = np.zeros((self.data.shape[0], self.nt))
        self.out[:, 0] = self.Init
        for t in range(self.nt-1):
            epsilon = np.zeros((self.data.shape[0]))
            if stochastic:
                for _ in range(self.data.shape[0]):
                    epsilon[_] = random.choice(self.e[_, :])
            self.out[:, t+1] = self._predict(self.out[:, t], epsilon)