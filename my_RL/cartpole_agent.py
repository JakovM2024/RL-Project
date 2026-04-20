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
    def __init__(self, learning_rate = .001, gamma = .99, entropy_coef = .01, clip_epsilson = .1, epochs = 4, gae_lamda = .95, value_coef = .5):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.clip_epsilson = clip_epsilson
        self.epochs = epochs
        self.gae_lamda = gae_lamda
        self.value_coef = value_coef
        self.actor = Actor()
        self.critic = Critic()

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr = learning_rate)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_prob = self.actor.forward(state_tensor)
        distribution = torch.distributions.Categorical(action_prob)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        value = self.critic(state_tensor)
        return action.item(), log_prob.item(), value.item()
    
    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, last_value):
        advantages = []
        gae = 0
        values = self.values + [last_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lamda * (1 - self.dones[t]) * gae #unclear what done satnds for
            advantages.insert(0,gae)
        
        returns = [adv + val for adv, val in zip(advantages , self.values)]
        return advantages, returns
 
    
    def update(self, last_value = 0): 
        advantages, returns = self.compute_gae(last_value)

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions) 
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            action_probs = self.actor(states)
            distribution =  torch.distributions.Categorical(action_probs)
            new_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()
            new_values = self.critic(states).squeeze()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilson, 1 + self.clip_epsilson) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss =  nn.MSELoss()(new_values, returns)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list (self.critic.parameters()), .5
            )
            self.optimizer.step()

        self.states, self.actions, self.dones = [], [], []
        self.rewards, self.values, self.log_probs = [], [], []

        



