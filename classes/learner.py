
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from .memory import ReplayMemory

import random
import math
import copy

from .cuda_setup import device
from .memory import Transition

class Learner():

    def __init__(self, network, constants, memory_size = 10000, optimizer = optim.RMSprop, loss_func = F.smooth_l1_loss):

        self.policy_net = copy.deepcopy(network).to(device)
        self.target_net = copy.deepcopy(network).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(memory_size)

        self.optimizer = optimizer(self.policy_net.parameters())
        self.loss_func = loss_func

        self.constants = constants
        if not isinstance(self.constants, dict):
            raise Exception("Constants must be a dict")

        self.steps_done = 0

    def select_action(self, state, increment = True):

        sample = random.random()
        eps_threshold = self.constants["EPS_END"] + (self.constants["EPS_START"] - self.constants["EPS_END"]) * \
            math.exp(-1. * self.steps_done / self.constants["EPS_DECAY"])
        if increment:
            self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.constants["BATCH_SIZE"]:
            #print("Increasing memory size still...")
            return
        transitions = self.memory.sample(self.constants["BATCH_SIZE"])
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch =  Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        #print("State action vals")
        #print(state_action_values)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.constants["BATCH_SIZE"], device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.constants["GAMMA"]) + reward_batch
        #print("Expected state action vals")
        #print(expected_state_action_values)


        # Compute Huber loss
        loss = self.loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        return
