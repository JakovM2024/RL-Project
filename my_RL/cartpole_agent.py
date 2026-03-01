import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size = 4, action_size =2, hidden_size=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
    def forward(self,state):
        return self.network(state)
    

class Critic(nn.Module):
    def __init__(self, state_size = 4, hidden_size=256):
        super(Critic,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )    
    def forward(self, state):
        return self.network(state)
    

class ActorCriticAgent:
    def __init__(self, learning_rate = .001, gamma = .99, entropy_coef = .01):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.actor = Actor()
        self.critic = Critic()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = learning_rate)
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_prob = self.actor.forward(state_tensor)
        distribution = torch.distributions.Categorical(action_prob)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob
    def update(self, state, reward, next_state, done, log_prob):
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        reward_tensor = torch.FloatTensor([reward])

        curr_value = self.critic(state_tensor)

        if done:
            target = reward_tensor
        else:
            target = reward_tensor + self.gamma * self.critic(next_state_tensor).detach()
        advantage = target - curr_value

        critic_loss = (advantage**2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_prob = self.actor.forward(state_tensor)
        distribution = torch.distributions.Categorical(action_prob)
        entropy = distribution.entropy()

        actor_loss = -log_prob * advantage.detach() - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()






            
        




