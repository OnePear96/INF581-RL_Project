import gym
import torch
from a3c.actor_critic import ActorCritic
from a3c.actions import get_action_space, get_actions
from a3c.environment_wrapper import EnvironmentWrapper


def actor_critic_inference(params, path):
    model = ActorCritic(params.stack_size, get_action_space())
    model.load_state_dict(torch.load(path))
    model.eval()

    game = params.game
    env = gym.make(game)
    env_wrapper = EnvironmentWrapper(env, params.stack_size)

    state = env_wrapper.reset()
    state = torch.Tensor([state])
    done = False
    total_score = 0
    while not done:
        probs, _, _ = model(state)
        action = get_actions(probs)
    #    print(action)
        state, reward, done = env_wrapper.step(action[0])
        state = torch.Tensor([state])
        total_score += reward
        env_wrapper.render()
    return total_score
