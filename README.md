# DeepRL-Pytorch

This is a project to document experimenting with DeepRL algorithms. The goal is to start with environments from OpenAI Gym and then move on to more advanced problems.

The inspiration for this repository came from trying to learn about DeepRL and all code examples I found were somewhat long, unorganized, and very script-y. The ultimate goal for this project is throughout experimentation and reading research papers, that we form a nice layer of abstraction to much easier prototype solutions to DeepRL problems similar to how FastAI and Keras did so for general deep learning applications.

The library is in extremely early stages so expect the api of accessing learning elements to change significantly as this project progresses.

## Components

The main components as of right now that are involved in the testing of these DeepRL problems are:

- GymEnv: The environment generator. Right now just wrapping OpenAI Gym environments.
- DQNImage: A neural network that will represent our policy and target networks
- ReplayMemory: A cache that fills with previous state / action / reward tuples and cycles in new events as time goes on. Is used for our learner to sample randomly from in order to increase stability when training and updating gradients.
- Learner: An abstraction to hold information for the different types of learning. This handles keeping track of information related to our policy and target networks as well as details of the learning process and updating gradients.
- Simulator: This is the main learning loop that will house the different strategies for solving DeepRL problems. Right now, it is just using Deep Q Learning. Will be expanded to employ different strategies.

*These Components are subject to change based on further experimentation.*

## Contributing

Please reach out if you are interested in contributing to this project (mark.k.hoffmann@jpl.caltech.edu / markkhoffmann@gmail.com). Some of the items that I am looking to add in the near future are items such as:

- Integrating TensorboardX for better logging (https://github.com/lanpa/tensorboardX)
- Different types of Learning methods beyond DeepQ (suggestions?)
- Optimizations for faster learning such as strategies used in this blog post (https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55)
- Incorporating ideas of "super convergence" as noted in FastAI such as cyclical learning rates
- Support for different types of core problems (games vs. scheduling vs. ...)
