from itertools import count
import torch

from .cuda_setup import device
from tqdm import tqdm_notebook as tqdm

from abc import ABC, abstractmethod

from .logger import log

class SimBase(ABC):

    def __init__(self, *args, **kwargs):
        super(SimBase, self).__init__()
        self.env = args[0]
        self.learner = args[1]
        self.TARGET_UPDATE = kwargs["TARGET_UPDATE"]

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self) -> bool:
        """
        Every simulator that inherits from here must:
            1.) set self.action based on self.state
            2.) take a step in their environment based on self.action
            3.) have self.reward, self.state, self.next_state update properly
            4.) return the done variable
        """
        pass

    def record(self):
        self.learner.memory.push(self.state, self.action, self.next_state, self.reward)

    def transition(self):
        # Move to the next state
        self.state = self.next_state


class Image2DSimulator(SimBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def initialize(self):
        state = self.env.env.reset()

        self.last_screen = self.env.get_screen()
        self.current_screen = self.env.get_screen()
        self.state = self.current_screen - self.last_screen
        return

    def step(self) -> bool:
        self.action = self.learner.select_action(self.state)
        _, reward, done, _ = self.env.env.step(self.action.item())
        self.reward = torch.tensor([reward], device=device)

        self.last_screen = self.current_screen
        self.current_screen = self.env.get_screen()
        if not done:
            self.next_state = self.current_screen - self.last_screen
        else:
            self.next_state = None
        return(done)


class StateSpaceSimulator(SimBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self):
        state = self.env.env.reset()

        self.state = torch.tensor([state], device=device).float()
        return

    def step(self) -> bool:
        self.action = self.learner.select_action(self.state)
        state, reward, done, _ = self.env.env.step(self.action.item())
        self.reward = torch.tensor([reward], device=device)
        self.state = torch.tensor([state], device=device).float()

        if not done:
            self.next_state = self.state
            self.env.env.render(mode='rgb_array').transpose(
                (2, 0, 1))
        else:
            self.next_state = None
        return(done)


class Simulator():

    def __init__(self, env, learner, TARGET_UPDATE = 10):

        if learner.policy_net.INPUT_TYPE == "IMAGE2D":
            self.simulator = Image2DSimulator(env, learner, TARGET_UPDATE=TARGET_UPDATE)
        elif learner.policy_net.INPUT_TYPE == "STATESPACE":
            self.simulator = StateSpaceSimulator(env, learner, TARGET_UPDATE=TARGET_UPDATE)

        self.env = self.simulator.env
        self.TARGET_UPDATE = TARGET_UPDATE
        self.episode_durations = []
        self.tot_episodes = 0

    def simulate(self, num_episodes):
        for i_episode in tqdm(range(num_episodes)):
            self.tot_episodes += 1
            log.info(f"Episode: {self.tot_episodes}")

            #Initialize the base env back
            self.simulator.initialize()

            for t in count():
                log.debug(f"Step: {t}")
                done = self.simulator.step()
                self.simulator.record()
                self.simulator.transition()
                # Perform one step of the optimization for policy net
                self.simulator.learner.optimize_model()

                if done:
                    self.episode_durations.append(t + 1)
                    break

            # Update the target network
            if i_episode % self.TARGET_UPDATE == 0:
                self.simulator.learner.update_target_net()

        self.env.env.render()
        self.env.env.close()
