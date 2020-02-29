import gym
import torch
from a3c.actor_critic import ActorCritic
from a3c.actions import get_action_space, get_actions
from a3c.environment_wrapper import EnvironmentWrapper


def evaluate_actor_critic(params, path):
    model = ActorCritic(params.stack_size, get_action_space())
    model.load_state_dict(torch.load(path))
    model.eval()

    game = params.game
    env = gym.make(game)
    env_wrapper = EnvironmentWrapper(env, params.stack_size)

    total_reward = 0
    num_of_episodes = 100

    for episode in range(num_of_episodes):
        state = env_wrapper.reset()
        state = torch.Tensor([state])
        done = False
        score = 0
        while not done:
            probs, _, _ = model(state)
            action = get_actions(probs)
            state, reward, done = env_wrapper.step(action[0])
            state = torch.Tensor([state])
            score += reward
            env_wrapper.render()
        print('Episode: {0} Score: {1:.2f}'.format(episode, score))
        total_reward += score
    return total_reward / num_of_episodes
