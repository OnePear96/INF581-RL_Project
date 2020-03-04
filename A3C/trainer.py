import torch
import multiprocessing as mp

import torch.optim as optim
import torch.nn.functional as F
from player import Player
from collections import deque
from runner import Runner
import itertools
import time
import traceback
class Trainer:
    def __init__(self, gamma, agent, window = 15, workers = 8, **kwargs):
        super().__init__(**kwargs)

        self.agent = agent
        self.window = window
        self.gamma = gamma
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-4)
        self.workers = workers

        # even though we're loading the weights into worker agents explicitly, I found that still without sharing the weights as following, the algorithm was not converging:
        self.agent.share_memory()

    def fit(self, episodes = 1000):
        """
            The higher level method for training the agents.
            It called into the lower level "train" which orchestrates the process itself.
        """
        last_update = 0
        updates = dict()

        for ix in range(1, self.workers + 1):
            updates[ ix ] = { 'episode': 0, 'step': 0, 'rewards': deque(), 'losses': deque(), 'points': 0, 'mean_reward': 0, 'mean_loss': 0 }

        for update in self.train(episodes):
            now = time.time()

            # you could do something useful here with the updates dict.
            # I've opted out as I'm using logging anyways and got more value in just watching the log file, grepping for the desired values

            # save the current model's weights every minute:
            if now - last_update > 60:
                torch.save(self.agent.state_dict(), './checkpoints/car-racing/' + str(int(now)) + '-.pytorch')
                last_update = now

    def train(self, episodes = 1000):
        """
        Lower level training orchestration method. Written in the generator style. Intended to be used with "for update in train(...):"
        """

        # create the requested number of background agents and runners:
        worker_agents = self.agent.clone(num = self.workers)
        runners = [ Runner(agent=agent, ix = ix + 1, train = True) for ix, agent in enumerate(worker_agents) ]

        # we're going to communicate the workers' updates via the thread safe queue:
        queue = mp.SimpleQueue()

        # if we've not been given a number of episodes: assume the process is going to be interrupted with the keyboard interrupt once the user (us) decides so:
        if episodes is None:
            print('Starting out an infinite training process')

        # create the actual background processes, making their entry be the train_one method:
        processes = [ mp.Process(target=self.train_one, args=(runners[ix - 1], queue, episodes, ix)) for ix in range(1, self.workers + 1) ]

        # run those processes:
        for process in processes:
            process.start()

        try:
            # what follows is a rather naive implementation of listening to workers updates. it works though for our purposes:
            while any([ process.is_alive() for process in processes ]):
                results = queue.get()
                yield results
        except Exception as e:
            print(str(e))

    def train_one(self, runner, queue, episodes = 1000, ix = 1):
        """
        Orchestrate the training for a single worker runner and agent. This is intended to run in its own background process.
        """

        # possibly naive way of trying to de-correlate the weight updates further (I have no hard evidence to prove if it works, other than my subjective observation):
        time.sleep(ix)

        try:
            # we are going to request the episode be reset whenever our agent scores lower than its max points. the same will happen if the agent scores total of -10 points:
            max_points = 0
            max_eval_points = 0
            min_points = 0
            max_episode = 0

            for episode_ix in itertools.count(start=0, step=1):

                if episodes is not None and episode_ix >= episodes:
                    return

                max_episode_points = 0
                points = 0

                # load up the newest weights every new episode:
                runner.load_state_dict(self.agent.state_dict())

                # every 5 episodes lets evaluate the weights we've learned so far by recording the run of the car using the greedy strategy:
                if ix == 1 and episode_ix % 5 == 0:
                    eval_points = self.record_greedy(episode_ix)

                    if eval_points > max_eval_points:
                        torch.save(runner.agent.state_dict(), './checkpoints/car-racing/' + str(eval_points) + '-eval-points.pytorch')
                        max_eval_points = eval_points

                # each n-step window, compute the gradients and apply
                # also: decide if we shouldn't restart the episode if we don't want to explore too much of the not-useful state space:
                for step, rewards, values, policies, action_ixs, terminal in runner.run_episode(yield_every=self.window):
                    points += sum(rewards)

                    if ix == 1 and points > max_points:
                        torch.save(runner.agent.state_dict(), './checkpoints/car-racing/' + str(points) + '-points.pytorch')
                        max_points = points

                    if ix == 1 and episode_ix > max_episode:
                        torch.save(runner.agent.state_dict(), './checkpoints/car-racing/' + str(episode_ix) + '-episode.pytorch')
                        max_episode = episode_ix

                    if points < -10 or (max_episode_points > min_points and points < min_points):
                        terminal = True
                        max_episode_points = 0
                        points = 0
                        runner.ask_reset()

                    if terminal:
                        print('TERMINAL for ' + str(ix) + ' at step ' + str(step) + ' with total points ' + str(points) + ' max: ' + str(max_episode_points) )

                    # if we're learning, then compute and apply the gradients and load the newest weights:
                    if runner.train:
                        loss = self.apply_gradients(policies, action_ixs, rewards, values, terminal, runner)
                        runner.load_state_dict(self.agent.state_dict())

                    max_episode_points = max(max_episode_points, points)
                    min_points = max(min_points, points)

                    # communicate the gathered values to the main process:
                    queue.put((ix, episode_ix, step, rewards, loss, points, terminal))

        except Exception as e:
            string = traceback.format_exc()
            print(str(e) + ' → ' + string)
            queue.put((ix, -1, -1, [-1], -1, str(e) + '<br />' + string, True))

    def record_greedy(self, episode_ix):
        """
        Records the video of the "greedy" run based on the current weights.
        """
        directory = './videos/car-racing/episode-' + str(episode_ix) + '-' + str(int(time.time()))
        player = Player(agent=self.agent, directory=directory, train=False)
        points = player.play()
        print('Evaluation at episode ' + str(episode_ix) + ': ' + str(points) + ' points (' + directory + ')')
        return points

    def apply_gradients(self, policies, actions, rewards, values, terminal, runner):
        worker_agent = runner.agent
        actions_one_hot = torch.tensor([[ int(i == action) for i in range(4) ] for action in actions], dtype=torch.float32)

        policies = torch.stack(policies)
        values = torch.cat(values)
        values_nograd = torch.zeros_like(values.detach(), requires_grad=False)
        values_nograd.copy_(values)

        discounted_rewards = self.discount_rewards(runner, rewards, values_nograd[-1], terminal)
        advantages = discounted_rewards - values_nograd

        print('Runner ' + str(runner.ix) + 'Rewards: ' + str(rewards))
        print('Runner ' + str(runner.ix) + 'Discounted Rewards: ' + str(discounted_rewards.numpy()))

        log_policies = torch.log(0.00000001 + policies)

        one_log_policies = torch.sum(log_policies * actions_one_hot, dim=1)

        entropy = torch.sum(policies * -log_policies)

        policy_loss = -torch.mean(one_log_policies * advantages)

        value_loss = F.mse_loss(values, discounted_rewards)

        value_loss_nograd = torch.zeros_like(value_loss)
        value_loss_nograd.copy_(value_loss)

        policy_loss_nograd = torch.zeros_like(policy_loss)
        policy_loss_nograd.copy_(policy_loss)

        print('Value Loss: ' + str(float(value_loss_nograd)) + ' Policy Loss: ' + str(float(policy_loss_nograd)))

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        self.agent.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(worker_agent.parameters(), 40)

        # the following step is crucial. at this point, all the info about the gradients reside in the worker agent's memory. We need to "move" those gradients into the main agent's memory:
        self.share_gradients(worker_agent)

        # update the weights with the computed gradients:
        self.optimizer.step()

        worker_agent.zero_grad()
        return float(loss.detach())

    def share_gradients(self, worker_agent):
        for param, shared_param in zip(worker_agent.parameters(), self.agent.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def clip_reward(self, reward):
        """
        Clips the rewards into the <-3, 3> range preventing too big of the gradients variance.
        """
        return max(min(reward, 3), -3)

    def discount_rewards(self, runner, rewards, last_value, terminal):
        discounted_rewards = [0 for _ in rewards]
        loop_rewards = [ self.clip_reward(reward) for reward in rewards ]

        if terminal:
            loop_rewards.append(0)
        else:
            loop_rewards.append(runner.get_value())

        for main_ix in range(len(discounted_rewards) - 1, -1, -1):
            for inside_ix in range(len(loop_rewards) - 1, -1, -1):
                if inside_ix >= main_ix:
                    reward = loop_rewards[inside_ix]
                    discounted_rewards[main_ix] += self.gamma**(inside_ix - main_ix) * reward

        return torch.tensor(discounted_rewards)