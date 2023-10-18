import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetyplusplus_folder.networks import Critic,Actor ,Encoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        rew_discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        expl_noise = 0.1,
    ):
        local_feature_map_in_channels=5
        example_encoder=Encoder(local_feature_map_in_channels)
        a=np.zeros([1,local_feature_map_in_channels,23,23])
        example_encoder=example_encoder.float()
        self.local_feature_map_out_channels=example_encoder(torch.tensor(a).float()).shape[1]


        self.actor = Actor(state_dim,local_feature_map_in_channels,self.local_feature_map_out_channels, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim,local_feature_map_in_channels,self.local_feature_map_out_channels, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        torch.set_num_threads(10)
        self.action_dim = action_dim
        self.max_action = max_action
        self.rew_discount = rew_discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.expl_noise = expl_noise
        self.total_it = 0
        


    def select_action(self,all_state,exploration=False):
        global_state=torch.FloatTensor(all_state[0]).unsqueeze(0).to(device)
        local_state=torch.FloatTensor(all_state[1]).unsqueeze(0).to(device)
        action = self.actor([global_state,local_state]).cpu().data.numpy().flatten()
        if exploration:
            noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action


    def train(self, replay_buffer, batch_size=256,train_nums=200):
        for _ in range(train_nums):
            self.total_it+=1
            # Sample replay buffer 
            global_state,local_state, action, next_global_state,next_local_state, reward, not_done = replay_buffer.sample(batch_size)
            # import pdb;pdb.set_trace()
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                next_action = (
                    self.actor_target([next_global_state,next_local_state]) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target([next_global_state,next_local_state], next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.rew_discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic([global_state,local_state], action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:

                # Compute actor loss
                action = self.actor([global_state,local_state])
                actor_loss = - self.critic.Q1([global_state,local_state], action).mean()
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            # import pdb;pdb.set_trace()
            
            
            
    def save(self, filename):
        torch.save(self.critic.state_dict(), f"{filename}_critic")
        torch.save(self.critic_optimizer.state_dict(), f"{filename}_critic_optimizer")
 
        torch.save(self.actor.state_dict(), f"{filename}_actor")
        torch.save(self.actor_optimizer.state_dict(), f"{filename}_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(f"{filename}_critic",map_location=lambda storage, loc: storage.cuda(0)))
        self.critic_optimizer.load_state_dict(torch.load(f"{filename}_critic_optimizer",map_location=lambda storage, loc: storage.cuda(0)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(f"{filename}_actor",map_location=lambda storage, loc: storage.cuda(0)))
        self.actor_optimizer.load_state_dict(torch.load(f"{filename}_actor_optimizer",map_location=lambda storage, loc: storage.cuda(0)))
        self.actor_target = copy.deepcopy(self.actor)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
# def eval_policy(policy, eval_env, seed, flag, eval_episodes=5):
#     avg_reward = 0.
#     avg_cost = 0.
#     for _ in range(eval_episodes):
#         if flag == 'constraint_violation':
#             reset_info, done = eval_env.reset(), False
#             state = reset_info[0]
#         else:
#             state, done = eval_env.reset(), False
#         while not done:
#             action = policy.select_action(np.array(state))
#             state, reward, done, info = eval_env.step(action)
#             avg_reward += reward
#             if info[flag]!=0:
#                 avg_cost += 1

#     avg_reward /= eval_episodes
#     avg_cost /= eval_episodes

#     print("---------------------------------------")
#     print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} Cost {avg_cost:.3f}.")
#     print("---------------------------------------")
#     return avg_reward,avg_cost