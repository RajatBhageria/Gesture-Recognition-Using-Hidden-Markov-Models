class HMM():
    def __init__(self, n_states, n_obs, Pi=None, A=None, B=None):

    def log_forward(self, obs_sequence):
        NotImplementedError

    def log_backward(self, obs_sequence):
        NotImplementedError

    def baum_welch(self, obs_sequence_list, max_iter=100):
        NotImplementedError

