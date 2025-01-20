import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym

# Hyperparameters
EPSILON = 0.2
GAMMA = 0.99
LAMBDA = 0.8
ENTROPY_COEF = 1e-3
BATCH = 128
EPOCH = 20
LEARNING_RATE = 3e-4
WEIGHTING_FACTOR = 0.5
GRAD_CLIP_THRESH = 0.5

# Initialize environment
env = gym.make("BipedalWalker-v3", hardcore=False, render_mode=None)
state_size = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]


# Actor-Critic neural network
class ActorCritic(nn.Module):
    def __init__(self, state_dim=state_size, action_dim=num_actions, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.log = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.actor(x)
        stdv = torch.exp(self.log)
        action_distribution = Normal(mean, stdv)
        state_value = self.critic(x)
        return action_distribution, state_value


# PPO agent
class PPO:
    def __init__(self):
        self.actor_critic = ActorCritic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LEARNING_RATE)
    
    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            action_distribution, state_value = self.actor_critic(state)
            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action).sum().item()
            state_value = state_value.item()
        return action.numpy(), log_prob, state_value
    
    def adr(self, rewards, values, dones):
        # Compute advantages and returns using GAE
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            next_value = values[t + 1] 
            delta = rewards[t] + GAMMA * next_value * (1 - int(dones[t])) - values[t]
            advantages[t] = delta + GAMMA * LAMBDA * advantages[t + 1] * (1 - int(dones[t]))
            returns[t] = rewards[t] + GAMMA * returns[t + 1] * (1 - int(dones[t]))

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6) # Normalize advantages
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def train(self, states, actions, rewards, values, log_probs, dones):
        # Convert lists to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        log_probs = torch.tensor(log_probs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        advantages, returns = self.adr(rewards, values, dones)
        for _ in range(EPOCH):
            action_distribution, critic_value = self.actor_critic(states)
            entropy = action_distribution.entropy().mean()  # Entropy for exploration
            new_log_probs = action_distribution.log_prob(actions)  # New log probabilities
            
            # Handle multi-dimensional actions
            if new_log_probs.dim() > 1:
                new_log_probs_sum = new_log_probs.sum(dim=1)
            else:
                new_log_probs_sum = new_log_probs
            
            if log_probs.dim() > 1:
                log_probs_sum = log_probs.sum(dim=1)
            else:
                log_probs_sum = log_probs
            
            # Compute probability ratio and PPO loss
            prob_ratio = torch.exp(new_log_probs_sum - log_probs_sum)
            weighted_probs = advantages * prob_ratio
            weighted_clipped_probs = advantages * torch.clamp(prob_ratio, 1 - EPSILON, 1 + EPSILON)
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean() - ENTROPY_COEF * entropy
            critic_loss = ((critic_value.squeeze() - returns) ** 2).mean()
            total_loss = actor_loss + WEIGHTING_FACTOR * critic_loss

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), GRAD_CLIP_THRESH)
            self.optimizer.step()

def main():
    # Initialize PPO agent
    agent = PPO()

    # Training loop
    n_episodes = 20000
    max_num_timesteps = 1000  # Maximum timesteps per episode
    num_p_av = 200  # Window size for average reward calculation
    total_point_history = []
    for i in range(n_episodes):
        state, _ = env.reset()
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        total_points = 0
        for _ in range(max_num_timesteps):
            action, log_prob, value = agent.choose_action(state) # Choose action
            new_state, reward, terminated, truncated, _ = env.step(action) # Take action
            done = terminated or truncated

            # Collect data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob) 
            dones.append(terminated)

            state = new_state.copy()
            total_points += reward

            if done:
                break

        # Train on collected data
        agent.train(states, actions, rewards, values, log_probs, dones)

        # Calculate and print average point 
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")
        if (i+1) % num_p_av == 0:
            print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")
        
        # Early stopping
        if av_latest_points > 200:
            break

    env.close()


main()