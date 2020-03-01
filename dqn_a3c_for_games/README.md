# car-racing-rl

Implementation of Reinforcement Learning algorithms in [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/)
[Breakout-ram-v0](https://gym.openai.com/envs/Breakout-ram-v0/)
[Atlantis-v0](https://gym.openai.com/envs/Atlantis-v0/)
environments. Implemented algorithms:
* Deep Q-Network (DQN)
* Asynchronous Advantage Actor Critic (A3C)

In which training on images for Atari games is done in A3C code, RAM in DQN code.

We can also use this code for the Mario game

Have fun

## Setup and running ##

Requirements: `python 3.6`

To install all required dependencies run:
```bash
pip install -r requirements.txt
```

Start training / inference / evaluation with:
```bash
python -m run --<action> -m=<model>
```
Possible values for parameter `action` are: `train`, `inference` and `evaluate`.

Possible values for parameter `model` are: `dqn`, `a3c`.

Hyperparameters can be changed in .json files in `/params` directory.