from agent import Agent
from trainer import Trainer
if __name__ == "__main__":
    agent = Agent()

    trainer = Trainer(gamma = 0.99, agent = agent)
    trainer.fit(episodes=None)