import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_of_inputs, num_of_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(num_of_inputs, 16, kernel_size=5, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
        self.batch_norm3 = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(2016, 256) #32*7*7
        self.policy = nn.Linear(256, num_of_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))
        conv2_out = F.relu(self.batch_norm2(self.conv2(conv1_out)))
        conv3_out = F.relu(self.batch_norm3(self.conv3(conv2_out)))

        flattened = torch.flatten(conv3_out, start_dim=1)  # N x 7*7*32
    #    print ("flatten shape:",flattened.size())
        linear1_out = self.linear1(flattened)

        policy_output = self.policy(linear1_out)
        value_output = self.value(linear1_out)

        probs = F.softmax(policy_output)
        log_probs = F.log_softmax(policy_output)
        return probs, log_probs, value_output


if __name__ == '__main__':
    x = torch.rand(1, 5, 84, 84)
    ac = ActorCritic(5, 5)
    ac(x)
