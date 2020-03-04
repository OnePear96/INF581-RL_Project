from runner import Runner
from gym.wrappers import Monitor

class Player(Runner):
    def __init__(self, directory, **kwargs):
        super().__init__(ix=999, **kwargs)

        self.env = Monitor(self.env, directory)

    def play(self):
        points = 0
        for step, rewards, values, policies, actions, terminal in self.run_episode(yield_every = 1, do_render = True):
            points += sum(rewards)
        self.env.close()
        return points