import numpy as np
import math
from scipy.misc import logsumexp

class HMM():
    def __init__(self, n_states, n_obs, Pi=None, A=None, B=None):
        self.n_states = n_states
        self.n_obs = n_obs
        self.Pi = Pi
        self.A = A
        self.B = B

    def log_forward(self, obs_sequence):
        [T,] = np.array(obs_sequence).shape
        alpha = np.zeros((self.n_states,T))

        #Initalization
        obsZero = obs_sequence[0]
        for i in range(0,self.n_states):
            alpha[i,0] = np.log(self.Pi[i]) + np.log(self.B[obsZero,i])

        #Induction
        for t in range(0,T-1):
            obsTP1 = obs_sequence[t+1]
            for j in range(0,self.n_states):
                AIJ = np.log(self.A[:,j])
                alphaI = alpha[:,t]
                sumOldAlphas = logsumexp(AIJ + alphaI)
                alpha[j,t+1] = sumOldAlphas + np.log(self.B[obsTP1, j])
        #Termination
        logProbObservations = logsumexp(alpha[:,T-1])
        return logProbObservations, alpha

    def log_backward(self, obs_sequence):
        [T,] = np.array(obs_sequence).shape
        beta = np.zeros((self.n_states,T))

        #Initalization
        for i in range(0,self.n_states):
            beta[i,self.n_obs-1] = np.log(1)

        #induction
        for t in range(T-2,-1,-1): #from the second to last observation to the first observation
            obsTP1 = obs_sequence[t+1]
            for i in range(0,self.n_states):
                AIJ = np.log(self.A[i, :])
                BJ = np.log(self.B[obsTP1, :])
                betaT = beta[:, t + 1]
                totalSum = logsumexp(AIJ + BJ + betaT)
                beta[i,t] = totalSum
        return beta

    def baum_welch(self, obs_sequence_list, max_iter=100):
        counter = 0
        while (counter < max_iter):
            sequence = obs_sequence_list
            [T,] = np.array(sequence).shape
            [probObservations,logAlpha] = self.log_forward(sequence)
            logBeta = self.log_backward(sequence)

            ##Find xi [n_states x n_states x n_obs - 1]
            xi = np.empty((self.n_states,self.n_states,T-1))
            for t in range(0,T-1):
                obsTP1 = sequence[t+1]
                for i in range(0,self.n_states):
                    for j in range(0,self.n_states):
                        numerator = logAlpha[i,t] + np.log(self.A[i,j]) + np.log(self.B[obsTP1,j]) + logBeta[j,t+1]
                        xi[i,j,t] = numerator-probObservations

            ##Find Gamma [n_states x n_obs -1]
            gamma = np.empty((self.n_states,T-1))
            gamma = logsumexp(xi,axis=1)

            ##Find PiBar
            piBar = gamma[:,0]

            ##Find ABar
            #This is a [n_state x n_state] matrix
            transitionsFromIToJ = logsumexp(xi,axis=2) #sum over time
            #This is a [n_state x 1] matrix
            transitionsFromI = logsumexp(gamma,axis=1) #sum over time
            ABar = transitionsFromIToJ - transitionsFromI

            ##Find B bar
            BBar = np.empty((self.B.shape))
            #Find the gamme function up to T (as opposed to to T-1)
            gammaUpToT = np.empty((self.n_states, T))
            for t in range(0, T):
                for i in range(0, self.n_states):
                    gammaUpToT[i, t] = (logAlpha[i, t] + logBeta[i, t]) - probObservations
            #Update BBar
            for k in range(0, self.n_obs):
                # Find numerator
                expectedTimesStateJAndObservingVk = np.zeros((self.n_states,))
                for t in range(0, T):
                    # add to total only if observed is the same as the row of the B matrix
                    observationT = sequence[t]
                    if observationT == k:
                        expectedTimesStateJAndObservingVk = expectedTimesStateJAndObservingVk + np.exp(gammaUpToT[:, t])
                expectedTimesStateJAndObservingVk = np.log(expectedTimesStateJAndObservingVk)
                # find denominator
                expectedTimesStateJ = logsumexp(gammaUpToT, axis=1)
                BBar[k, :] = (expectedTimesStateJAndObservingVk - expectedTimesStateJ).T

            ##Set A = ABar, B = Bar, and Pi = PiBar
            self.A = np.exp(ABar)
            self.B = np.exp(BBar)
            self.Pi = np.exp(piBar)

            #increase counter by one for the next iteration
            counter = counter + 1

        return self.A, self.B, self.Pi