import numpy as np

class HMM():
    def __init__(self, n_states, n_obs, Pi=None, A=None, B=None):
        self.n_states = n_states
        self.n_obs = n_obs
        self.Pi = Pi
        self.A = A
        self.B = B

    def log_forward(self, obs_sequence):
        alpha = np.zeros((self.n_states,self.n_obs))

        #Initalization
        for i in range(0,self.n_states):
            val = np.log(self.Pi[i]) + np.log(self.B[0,i])
            alpha[i,0] = val

        #Induction
        for obs in range(0,self.n_obs-1):
            for j in range(0,self.n_states):
                sumOldAlphas = 0
                for i in range(0, self.n_states):
                    alphaI = alpha[i,obs]
                    AIJ = np.log(self.A[i, j])
                    sum = alphaI + AIJ
                    sumOldAlphas = sumOldAlphas + np.exp(sum)
                alpha[j,obs+1] = np.log(sumOldAlphas) + np.log(self.B[obs+1, j])

        #Termination
        logProbObservations = np.sum(alpha[:,self.n_obs-1])
        return logProbObservations, alpha


    def log_backward(self, obs_sequence):
        beta = np.zeros((self.n_states,self.n_obs))

        #Initalization
        for i in range(0,self.n_states):
            beta[i,self.n_obs-1] = np.log(1)

        #induction
        for obs in range(self.n_obs-2,-1,-1): #from the second to last observation to the first observation
            for i in range(0,self.n_states):
                totalSum = 0
                for j in range(0, self.n_states):
                    AIJ = np.log(self.A[i,j])
                    BJ = np.log(self.B[obs+1,j])
                    betaT = beta[j,obs+1]
                    sum = AIJ + BJ + betaT
                    totalSum = totalSum + np.exp(sum)
                beta[i,obs] = np.log(totalSum)

        return beta


    def baum_welch(self, obs_sequence_list, max_iter=100):
        counter = 0
        while (counter < max_iter):
            [probObservations,logAlpha] = self.log_forward(obs_sequence_list)
            logBeta = self.log_backward(obs_sequence_list)

            ##Find xi [n_states x n_states x n_obs - 1]
            xi = np.empty((self.n_states,self.n_states,self.n_obs-1))
            for t in range(0,self.n_obs-1):
                for i in range(0,self.n_states):
                    for j in range(0,self.n_states):
                        numerator = logAlpha[i,t] + np.log(self.A[i,j]) + np.log(self.B[t+1,j]) + logBeta[j,t+1]
                        denominator = probObservations
                        xi[i,j,t] = numerator/denominator

            ##Find Gamma [n_states x n_obs -1]
            gamma = np.empty((self.n_states,self.n_obs-1))
            gamma = np.sum(xi,axis=1)

            ##Find PiBar
            piBar = gamma[:,0]
            #normalize
            piBar = piBar / np.linalg.norm(piBar)

            ##Find ABar
            #This is a [n_state x n_state] matrix
            transitionsFromIToJ = np.sum(xi,axis=2) #sum over time
            #This is a [n_state x 1] matrix
            transitionsFromI = np.sum(gamma,axis=1) #sum over time
            ABar = transitionsFromIToJ / transitionsFromI

            #Find B bar
            #find the new gamma for gamma up to time t
            BBar = np.empty((self.B.shape))
            gammaUpToT = np.empty((self.n_states, self.n_obs))
            for t in range(0,self.n_obs):
                for i in range(0,self.n_states):
                    gammaUpToT[i,t] = (logAlpha[i,t] + logBeta[i,t])/probObservations

            for k in range(0,self.n_obs):
                # find numerator
                expectedTimesStateJAndObservingVk = np.zeros((self.n_states,))
                for t in range(0,self.n_obs):
                    expectedTimesStateJAndObservingVk = expectedTimesStateJAndObservingVk + gammaUpToT[:,t]
                #find denominator
                expectedTimesStateJ = np.sum(gammaUpToT, axis=1)
                BBar[k,:] = (expectedTimesStateJAndObservingVk / expectedTimesStateJ).T

            ##Set A = ABar, B = Bar, and Pi = PiBar
            self.A = np.exp(ABar)
            self.B = np.exp(BBar)
            self.Pi = np.exp(piBar)

            #increase counter by one for the next iteration
            counter = counter + 1

        return np.exp(self.A), np.exp(self.B), np.exp(self.Pi)