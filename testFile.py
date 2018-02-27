import numpy as np
import pickle
from HMM import HMM

# number of hidden states N
n_states = 10

# number of observation types M
n_obs = 30

# instantiate variables
pi = (1.0 / n_states) * np.ones((n_states, 1))

# A and B matrix
A = np.random.rand(n_states, n_states)
A = A / A.sum(axis=1)[:, None]

B = np.random.rand(n_obs, n_states)
B = B / B.sum(axis=1)[:, None]

with open('waveObs.pickle', 'rb') as handle:
    observationSequences = pickle.load(handle)
with open('beat4Obs.pickle', 'rb') as handle:
    test = pickle.load(handle)
# Generate the trained HMM model for the correct gesture
hmmModelOfGesture = HMM(n_states, n_obs, pi, A, B)
observationSequence = observationSequences[0]
[A, B, pi] = hmmModelOfGesture.baum_welch(observationSequence,max_iter=3)
print np.sum(A[:,9])
print np.sum(B[:,0])
print np.sum(pi)
[prob, alpha] = hmmModelOfGesture.log_forward(observationSequence)
print prob
[prob, alpha] = hmmModelOfGesture.log_forward(test[0])
print prob