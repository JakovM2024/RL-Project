"""
agent.py — The Actor-Critic Agent
===================================

This file creates the AGENT that learns to navigate the grid world.

We use an ACTOR-CRITIC architecture, which has two parts:

1. THE ACTOR (the "decision maker")
   - Looks at the current state and decides what action to take.
   - Outputs PROBABILITIES for each action (e.g., 30% up, 50% right, 10% down, 10% left).
   - This probability distribution is called the POLICY.
   - Think of it as: "Given where I am, what should I probably do?"

2. THE CRITIC (the "evaluator")
   - Looks at the current state and estimates how good it is.
   - Outputs a single number called the VALUE.
   - Think of it as: "Given where I am, how much total reward can I expect?"

WHY TWO NETWORKS?
   - The critic helps the actor learn faster by providing a "baseline".
   - Instead of the actor asking "was that action good?", it asks
     "was that action BETTER THAN EXPECTED?" — this is called the ADVANTAGE.
   - Advantage = actual reward - expected reward (from critic)
   - If advantage > 0: the action was better than expected → do it more!
   - If advantage < 0: the action was worse than expected → do it less!

Both networks are small neural networks built with PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    """
    The ACTOR network — learns the POLICY (which actions to take).

    POLICY: A mapping from states to action probabilities.
    For example, if the agent is near the goal, the policy might say
    "90% chance move toward goal, 10% chance move elsewhere".

    Architecture:
        Input (2) → Hidden (64, ReLU) → Output (4, Softmax)
        - Input: the state [row, col] normalized to 0-1
        - Hidden: 64 neurons with ReLU activation (introduces non-linearity)
        - Output: probability for each of the 4 actions (must sum to 1)
    """

    def __init__(self, state_size=2, action_size=4, hidden_size=64):
        """
        Create the actor network.

        Args:
            state_size: Number of values in the state (2: row and col).
            action_size: Number of possible actions (4: up/down/left/right).
            hidden_size: Number of neurons in the hidden layer.
        """
        super(Actor, self).__init__()

        # The neural network layers
        # nn.Linear = a fully connected layer (every input connects to every output)
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),   # Input → Hidden
            nn.ReLU(),                             # Activation function
            nn.Linear(hidden_size, action_size),   # Hidden → Output
            nn.Softmax(dim=-1)                     # Convert to probabilities (sum to 1)
        )

    def forward(self, state):
        """
        Given a state, output action probabilities.

        This is called the "forward pass" — data flows forward through the network.

        Args:
            state: A tensor of shape [2] — the agent's position.

        Returns:
            action_probs: A tensor of shape [4] — probability for each action.
        """
        return self.network(state)


class Critic(nn.Module):
    """
    The CRITIC network — learns the VALUE FUNCTION (how good is each state).

    VALUE FUNCTION: Estimates the total future reward from a given state.
    For example, if the agent is one step from the goal, the value should
    be high (~10). If it's far away, the value should be lower.

    Architecture:
        Input (2) → Hidden (64, ReLU) → Output (1)
        - Input: the state [row, col] normalized to 0-1
        - Hidden: 64 neurons with ReLU activation
        - Output: a single number (the estimated value of this state)
    """

    def __init__(self, state_size=2, hidden_size=64):
        """
        Create the critic network.

        Args:
            state_size: Number of values in the state (2: row and col).
            hidden_size: Number of neurons in the hidden layer.
        """
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),  # Input → Hidden
            nn.ReLU(),                            # Activation function
            nn.Linear(hidden_size, 1)             # Hidden → single value output
        )

    def forward(self, state):
        """
        Given a state, output the estimated value.

        Args:
            state: A tensor of shape [2] — the agent's position.

        Returns:
            value: A tensor of shape [1] — how good this state is.
        """
        return self.network(state)


class ActorCriticAgent:
    """
    The full Actor-Critic agent that combines both networks.

    This is the "brain" of our RL agent. It:
    1. Uses the ACTOR to decide which action to take.
    2. Uses the CRITIC to evaluate how good the current state is.
    3. After each step, updates both networks to get better over time.

    KEY HYPERPARAMETERS:
    - learning_rate: How big of a step to take when updating the networks.
                     Too high = unstable learning. Too low = slow learning.
    - gamma (discount factor): How much to value future rewards vs immediate rewards.
                               0.99 means future rewards are almost as important as now.
                               0.0 would mean only care about immediate reward.
    """

    def __init__(self, learning_rate=0.001, gamma=0.99):
        """
        Create the actor-critic agent.

        Args:
            learning_rate: Step size for network updates (default: 0.001).
            gamma: Discount factor for future rewards (default: 0.99).
        """
        self.gamma = gamma

        # Create the two networks
        self.actor = Actor()
        self.critic = Critic()

        # Create optimizers — these handle the math of updating network weights
        # Adam is a popular optimizer that adapts the learning rate automatically
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def select_action(self, state):
        """
        Choose an action based on the current state.

        The actor outputs probabilities for each action, and we SAMPLE
        from that distribution. This means the agent doesn't always pick
        the "best" action — it EXPLORES by sometimes trying other actions.

        This exploration is crucial! If the agent always picked the same
        action, it might never discover a better path.

        Args:
            state: numpy array of shape [2] — the agent's position.

        Returns:
            action: int — the chosen action (0-3).
            log_prob: tensor — log probability of the chosen action (needed for training).
        """
        # Convert numpy state to a PyTorch tensor
        state_tensor = torch.FloatTensor(state)

        # Get action probabilities from the actor
        action_probs = self.actor(state_tensor)

        # Create a categorical (discrete) probability distribution
        # This lets us SAMPLE an action based on the probabilities
        distribution = torch.distributions.Categorical(action_probs)

        # Sample an action from the distribution
        action = distribution.sample()

        # Get the log probability of this action (needed for the policy gradient)
        # Log probabilities are used because they're more numerically stable
        log_prob = distribution.log_prob(action)

        return action.item(), log_prob

    def update(self, state, reward, next_state, done, log_prob):
        """
        Update both networks after taking a step.

        This is where the LEARNING happens! Here's the process:

        1. The CRITIC estimates the value of the current state and next state.
        2. We compute the TD TARGET: reward + gamma * next_value (if not done).
           TD = "Temporal Difference" — learning from the difference between
           what we expected and what we got.
        3. The ADVANTAGE = TD target - current value estimate.
           Positive advantage = "that was better than expected!"
           Negative advantage = "that was worse than expected!"
        4. Update the CRITIC to be more accurate (minimize prediction error).
        5. Update the ACTOR based on the advantage:
           - If advantage > 0: increase the probability of this action.
           - If advantage < 0: decrease the probability of this action.

        Args:
            state: numpy array — the state before the action.
            reward: float — the reward received.
            next_state: numpy array — the state after the action.
            done: bool — whether the episode ended.
            log_prob: tensor — log probability of the action taken.
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        reward_tensor = torch.FloatTensor([reward])

        # --- STEP 1: Get value estimates from the critic ---
        current_value = self.critic(state_tensor)      # V(s)  — value of current state
        next_value = self.critic(next_state_tensor)     # V(s') — value of next state

        # --- STEP 2: Compute the TD target ---
        # If the episode is done, there's no future value
        # If not done, the target includes discounted future value
        if done:
            td_target = reward_tensor
        else:
            td_target = reward_tensor + self.gamma * next_value.detach()

        # --- STEP 3: Compute the advantage ---
        advantage = td_target - current_value
        # advantage > 0: action was better than expected
        # advantage < 0: action was worse than expected

        # --- STEP 4: Update the CRITIC ---
        # The critic's job is to be accurate, so we minimize the squared error
        # between its prediction (current_value) and the target (td_target)
        critic_loss = advantage.pow(2).mean()  # Mean squared error

        self.critic_optimizer.zero_grad()  # Clear old gradients
        critic_loss.backward()             # Compute new gradients
        self.critic_optimizer.step()       # Update critic's weights

        # --- STEP 5: Update the ACTOR ---
        # The actor's loss uses the POLICY GRADIENT theorem:
        # loss = -log_prob * advantage
        # The negative sign is because we want to MAXIMIZE reward,
        # but optimizers MINIMIZE loss, so we flip the sign.
        actor_loss = -log_prob * advantage.detach()
        # .detach() the advantage because we don't want actor updates
        # to flow back into the critic

        self.actor_optimizer.zero_grad()   # Clear old gradients
        actor_loss.backward()              # Compute new gradients
        self.actor_optimizer.step()        # Update actor's weights


# --- QUICK TEST ---
# If you run this file directly, it will create an agent and test action selection
if __name__ == "__main__":
    print("Testing the Actor-Critic Agent")
    print("=" * 40)

    agent = ActorCriticAgent()

    # Create a fake state (agent at position 0, 0)
    test_state = np.array([0.0, 0.0], dtype=np.float32)

    print(f"State: {test_state}")
    print(f"\nAction probabilities from untrained actor:")

    # Get action probabilities
    state_tensor = torch.FloatTensor(test_state)
    probs = agent.actor(state_tensor)
    for i, prob in enumerate(probs):
        action_name = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}[i]
        print(f"  {action_name}: {prob.item():.2%}")

    print(f"\nValue estimate from untrained critic:")
    value = agent.critic(state_tensor)
    print(f"  V(state) = {value.item():.4f}")

    print(f"\nSelecting an action...")
    action, log_prob = agent.select_action(test_state)
    action_name = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}[action]
    print(f"  Chose: {action_name} (log_prob: {log_prob.item():.4f})")
