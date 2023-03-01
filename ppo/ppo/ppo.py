
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import ppo.policy
import ppo.value_network
import random
import numpy as np
# import wandb

class PPO_Agent(nn.Module):
    def __init__(self, obs_space_size, action_space_size, num_steps,
                 epsilon=0.2, discount_gamma=0.99, gae_lambda=0.95, num_epochs=4,
                 batch_size=40):
        super(PPO_Agent, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._obs_space_size = obs_space_size
        self._action_space_size = action_space_size
        self._epsilon = epsilon
        self._discount_gamma = discount_gamma
        self._gae_lambda = gae_lambda
        self._num_epochs = num_epochs
        self._batch_size = batch_size

        self.state_data = torch.zeros((num_steps, obs_space_size)).to(self.device)
        self.action_data = torch.zeros((num_steps)).to(self.device)
        self.logprobs_data = torch.zeros((num_steps)).to(self.device)
        self.rewards_data = torch.zeros((num_steps)).to(self.device)
        self.dones_data = torch.zeros((num_steps)).to(self.device)
        self.value_ests_data = torch.zeros((num_steps)).to(self.device)

        self.actor = ppo.policy.Policy(action_space_size, obs_space_size, num_layers=2, num_hidden=256)
        self.critic = ppo.value_network.ValueNet(obs_space_size, num_layers=2, num_hidden=256)

    def get_action(self, current_state):
        logits = self.actor.forward(current_state)
        probs = Categorical(logits=logits)
        action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.get_value(current_state)
    
    def get_value(self, state):
        return self.critic(state)

    # start with very simple advantage function calculation and make it more complex later (GAE, etc)
    def store_rollout(self, step, state, action, log_probs, rewards, done, value_ests):
        self.state_data[step, :] = state
        self.action_data[step] = action
        self.logprobs_data[step] = log_probs
        self.rewards_data[step] =  rewards
        self.dones_data[step] = done
        self.value_ests_data[step] = value_ests

    def calc_advantage(self, next_state, next_done, num_steps):

        next_state = torch.tensor(next_state).to(self.device)
        with torch.no_grad():
            next_value = self.get_value(next_state).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards_data).to(self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones_data[t + 1]
                    nextvalues = self.value_ests_data[t + 1]
                delta = self.rewards_data[t] + self._discount_gamma * nextvalues * nextnonterminal - self.value_ests_data[t]
                advantages[t] = lastgaelam = delta + self._discount_gamma * self._gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.value_ests_data

            return returns, advantages


    def learn(self, num_steps, returns, advantages):        
        b_inds = np.arange(num_steps)
        clipfracs = []
        
        for e in range(self._num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, self._batch_size):
                end = start + self._batch_size
                batch_inds = b_inds[start:end]

                batch_states = self.state_data[batch_inds]
                batch_actions = self.action_data[batch_inds]
                batch_log_probs = self.logprobs_data[batch_inds]
                logits = self.actor(batch_states)
                probs = Categorical(logits=logits)
                new_log_probs = probs.log_prob(batch_actions)
                entropy = probs.entropy()
                new_value = self.get_value(batch_states)
                logratio = new_log_probs - batch_log_probs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self._epsilon).float().mean().item()]


                batch_advantages = advantages[batch_inds]
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                pg_loss1 = batch_advantages * ratio
                pg_loss2 = batch_advantages * torch.clamp(ratio, 1 - self._epsilon, 1 + self._epsilon)
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                batch_returns = returns[batch_inds]
                v_loss = 0.5 * ((new_value - batch_returns) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.2

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  ## TODO: maybe add this in?
                self.actor.optimizer.step()
                self.critic.optimizer.step()


                # wandb.log({'loss_clip': pg_loss})
                # wandb.log({'loss_value': v_loss})
                # wandb.log({'loss_entropy': entropy_loss})
                # wandb.log({'loss_total': loss})


