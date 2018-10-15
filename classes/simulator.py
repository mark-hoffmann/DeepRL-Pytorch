from itertools import count
import torch

from .cuda_setup import device
from tqdm import tqdm_notebook as tqdm

class Simulator():

    def __init__(self, env, learner, TARGET_UPDATE = 10):
        self.env = env
        self.learner = learner
        self.TARGET_UPDATE = TARGET_UPDATE

        self.episode_durations = []


    def simulate(self, num_episodes):
        for i_episode in tqdm(range(num_episodes)):
            #print("Episode: ", i_episode)
            # Initialize the environment and state
            self.env.env.reset()
            last_screen = self.env.get_screen()
            current_screen = self.env.get_screen()
            state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                action = self.learner.select_action(state)
                _, reward, done, _ = self.env.env.step(action.item())
                reward = torch.tensor([reward], device=device)

                # Observe new state
                last_screen = current_screen
                current_screen = self.env.get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                self.learner.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.learner.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    #plot_durations(episode_durations)
                    break
            # Update the target network
            if i_episode % self.TARGET_UPDATE == 0:
                self.learner.update_target_net()

        self.env.env.render()
        self.env.env.close()
