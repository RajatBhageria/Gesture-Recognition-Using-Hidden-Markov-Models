# ESE650 HMM project example model.
from HMM import HMM
import numpy as np

def generate_observations(model_name, T):
    """
    The Ride model from west Philly to Engineering.
    State : Chesnut St., Walnut St., Spruce St., Pine St.
    Observation : Students (five - S, W, P, W, C)
    model_name : name of a model
    T : length of a observation sequence to generate
    """
    if model_name == 'oober':
        A = np.array([[0.4, 0.4, 0.1, 0.1],
                      [0.3, 0.3, 0.3, 0.1],
                      [0.1, 0.3, 0.3, 0.3],
                      [0.1, 0.1, 0.4, 0.4]], dtype=np.float32)

        B = np.array([[0.0, 0.2, 0.0, 0.2, 0.6],
                      [0.1, 0.4, 0.0, 0.4, 0.1],
                      [0.5, 0.2, 0.1, 0.2, 0.0],
                      [0.3, 0.1, 0.5, 0.1, 0.0]], dtype=np.float32)

        Pi = np.array([0.3, 0.4, 0.1, 0.2], dtype=np.float32)

    elif model_name == 'nowaymo':
        A = np.array([[0.5, 0.1, 0.1, 0.3],
                      [0.2, 0.6, 0.1, 0.1],
                      [0.05, 0.1, 0.8, 0.05],
                      [0, 0.1, 0.2, 0.7]], dtype=np.float32)

        B = np.array([[0.0, 0.2, 0.0, 0.2, 0.6],
                      [0.1, 0.4, 0.0, 0.4, 0.1],
                      [0.5, 0.2, 0.1, 0.2, 0.0],
                      [0.3, 0.1, 0.5, 0.1, 0.0]], dtype=np.float32)

        Pi = np.array([0.2, 0.2, 0.1, 0.5], dtype=np.float32)

    elif model_name == 'dummy':
        A = np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        B = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

        Pi = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    state = inv_sampling(Pi)
    obs_sequence = []
    for t in xrange(T):
        obs_sequence.append(inv_sampling(B[state, :]))
        state = inv_sampling(A[state, :])
    return A, B, Pi, np.array(obs_sequence)


def inv_sampling(pdf):
    r = np.random.rand()
    for (i, p) in enumerate(np.cumsum(pdf)):
        if r <= p:
            return i

if __name__ == "__main__":
    [A, B, Pi, observations] = generate_observations('oober',100)
    n_states = A.shape[0]
    n_obs = 5
    HMM = HMM(n_states, n_obs, Pi.T, A, B.T)
    [A, newB, pi] = HMM.baum_welch(observations, max_iter=1)
    for i in range(0,100):
        [_,_,_, observationsOO] = generate_observations('oober',20)
        [_, _, _, observationsNo] = generate_observations('nowaymo',20)
        [probOO, _] = HMM.log_forward(observationsOO)
        [probNo, _] = HMM.log_forward(observationsNo)

        #print "oober" + str(probOO)
        #print "nowaymo" + str(probNo)