import numpy as np


class QEnv:
    def __init__(self, qcircuit, max_iter=200):
        self.qcircuit = qcircuit
        self.state = None
        self.iteration = 0
        self.max_iter = max_iter

    def _get_observation(self):
        observation = np.random.rand()
        return observation

    def step(self, action):
        self.iteration += 1

        """
        TODO: Apply action to the environment
        """

        self.state = self._get_observation()

        done = not (self.iteration < self.max_iter)
        done = bool(done)

        reward = 0
        if done:
            reward = 1

        return self.state, reward, done, {}
