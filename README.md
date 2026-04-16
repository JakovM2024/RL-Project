In this project I explored reinforcement learning methods, specifically Actor-Critic and PPO, through simulated 
experiments on an inverted pendulum and a discrete grid world. This project taught me how to use common libraries
like gymnasium and pytorch for the simulation and learning process of agents. The exterior directory hold my grid-world
agent which creates a 5x5 grid world enviorment with discrete actions(up, down, left, right) and trains an agent
to navigate from a start point to a goal with wall obstacles in between and without "a prior" knowledge of where the goal
is relative to the start position of the agent. The my_RL directory holds the inverted pendulum agent which utilizes PPO.
This agent learns to balance the pendulum without prior knowledge that an upright straight position is the ideal state,
it instead learns that this is the state it should seek through it's actions.
