import numpy as np

class HMM():
    def __init__(self, n_states, n_obs, Pi=None, A=None, B=None):
        self.n_states = n_states
        self.n_obs = n_obs
        self.Pi = Pi
        self.A = A
        self.B = B

    def forward(self, obs_sequence):
        alpha = np.zeros((self.n_states,self.n_obs))

        #Initalization
        for i in range(0,self.n_states):
            val = self.Pi[i]*self.B[0,i]
            alpha[i,0] = val

        #Induction
        for obs in range(1,self.n_obs):
            for j in range(0, self.n_states):
                sumOldAlphas = 0
                for i in range(0, self.n_states):
                    alphaI = alpha[i,obs]
                    AIJ = self.A[i, j]
                    sum = alphaI * AIJ
                    sumOldAlphas = sumOldAlphas + sum
                alpha[j,obs] = sumOldAlphas * self.B[obs, j]

        #Termination
        probObservations = np.sum(alpha[:,self.n_obs-1])
        return probObservations

    def log_forward(self, obs_sequence):
        alpha = np.zeros((self.n_states,self.n_obs))

        #Initalization
        for i in range(0,self.n_states):
            val = np.log(self.Pi[i]) + np.log(self.B[0,i])
            alpha[i,0] = val

        #Induction
        for obs in range(1,self.n_obs):
            for j in range(0, self.n_states):
                sumOldAlphas = 0
                for i in range(0, self.n_states):
                    alphaI = np.log(alpha[i,obs])
                    AIJ = np.log(self.A[i, j])
                    sum = alphaI + AIJ
                    sumOldAlphas = sumOldAlphas + np.exp(sum)
                alpha[j,obs] = np.log(sumOldAlphas) + np.log(self.B[obs, j])

        #Termination
        probObservations = np.sum(alpha[:,self.n_obs-1])
        return probObservations

    def backward(self, obs_sequence):
        beta = np.zeros((self.n_states,self.n_obs))

        #Initalization
        for i in range(0,self.n_states):
            beta[i,self.n_obs-1] = 1

        #induction
        for obs in range(self.n_obs-1,-1,-1): #from the second to last observation to the first observation
            for i in range(0,self.n_states):
                totalSum = 0
                for j in range(0, self.n_states):
                    AIJ = self.A[i,j]
                    BJ = self.B[obs,j]
                    betaT = beta[j,obs]
                    sum = AIJ * BJ * betaT
                    totalSum = totalSum + sum
                beta[i,obs] = totalSum


    def log_backward(self, obs_sequence):
        beta = np.zeros((self.n_states,self.n_obs))

        #Initalization
        for i in range(0,self.n_states):
            beta[i,self.n_obs-1] = np.log(1)

        #induction
        for obs in range(self.n_obs-1,-1,-1): #from the second to last observation to the first observation
            for i in range(0,self.n_states):
                totalSum = 0
                for j in range(0, self.n_states):
                    AIJ = np.log(self.A[i,j])
                    BJ = np.log(self.B[obs,j])
                    betaT = np.log(beta[j,obs])
                    sum = AIJ + BJ + betaT
                    totalSum = totalSum + np.exp(sum)
                beta[i,obs] = np.log(totalSum)

    def baum_welch(self, obs_sequence_list, max_iter=100):
        NotImplementedError