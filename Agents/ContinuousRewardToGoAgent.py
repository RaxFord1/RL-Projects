import gym
import pybulletgym
import tensorflow as tf
import numpy as np
import math

import utils.functions as functions

EPS = 1e-8

class DiscreteRewardToGoAgent:
    '''
    Agent with environment
    '''
    def __init__(self, env = None, learning_rate = 1e-2, seed = 0, sizes = [32], activation = tf.tanh, save = False):
        '''Initializing all specific variables for bot'''
        
        assert env, "Value env is required"
        
        self.env = gym.make(env)
        self.seed = 0
        self.obs = self.env.reset()
        self.activation = activation
        self.sizes = sizes
        self.obs_dim = self.env.observation_space.shape[0]
        self.n_acts = self.env.action_space.shape[0]
        self.learning_rate = learning_rate
        self.obs_ph, self.logits = functions.create_model(obs_dim = self.obs_dim, n_acts = self.n_acts, sizes = self.sizes, activation = self.activation)
        self.action = tf.squeeze(tf.multinomial(logits=self.logits, \
                                           num_samples=1), axis=1)
        self.weights_ph, self.act_ph, self.loss = functions.categorical_loss_func_weights(self.logits, self.n_acts)

        self.train_op = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
      
    def act(self, obs):
        '''Choosing action for self.obs '''
        action = self.sess.run(self.action, feed_dict = {
            self.obs_ph : [obs]
        })#[0] or [1]
        return action[0]
    
    
    def play_episode(self, rendering = False):
        '''
        Playing only one episode, collecting trajectory and rewards
        '''
        episode_obs_batch = list()
        episode_action_batch = list()
        episode_trajectory = list()
        rewards = list()
        
        self.obs = self.env.reset()
        while True:
            episode_obs_batch.append(self.obs.copy())
            if rendering == True:
                self.env.render()
            action = self.act(self.obs)
            episode_trajectory.append((self.obs, action))
            self.obs, reward, is_done, info = self.env.step(action)
            
            
            episode_action_batch.append(action)
            
            rewards.append(reward)
            if is_done:
                break
        if rendering == True:
            return sum(rewards)
        else:
            return rewards, episode_obs_batch, episode_action_batch, episode_trajectory
    
    
    def play_epoch(self, n):
        '''
        Collecting samples with current policy
        n = number of samples from 1 epoch
        '''
        epoch_obs_batch = list()
        epoch_action_batch = list()
        epoch_rewards = list()
        epoch_trajectory = list()
        epoch_reward_to_go = list()
        
        while len(epoch_obs_batch)<n:
            reward, obs_batch, action_batch, trajectory_batch = self.play_episode()
            
            epoch_obs_batch+=(obs_batch)
            
            epoch_action_batch+=(action_batch)
            epoch_rewards.append(sum(reward))
            epoch_reward_to_go += list(functions.reward_to_go(reward))
            
            epoch_trajectory.append(trajectory_batch)
        return epoch_obs_batch, epoch_action_batch, epoch_trajectory, epoch_reward_to_go, sum(epoch_rewards)/len(epoch_rewards)
    
    
    def train_epoch(self):
        '''
        Training agent for n epochs
        '''
        obs_batches, action_batches, trajectory_batches, reward_to_go, mean_reward = self.play_epoch(5000)
        self.play_episode(True)
        #print("obs: {}, \nactions: {},\n rewards: {}".format(obs_batches[0:8], action_batches[0:8], reward_to_go[0:8]))
        loss, _ = self.optimize(obs_batches, action_batches, reward_to_go)
        #print(np.array(obs_batches).shape,np.array(action_batches).shape,np.array(reward_to_go).shape)
        return loss, mean_reward
        
    def can_solve(self):
        for i in range(100):
            reward, _, _, _= self.play_episode()
            if sum(reward) != 200: 
                print("{} Cannot solve yet {}".format(i, sum(reward)))
                return False
        return True
                
        
    def train_n_epochs(self, n):
        for i in range(n):
            loss, reward = self.train_epoch()
            if reward == 200:
                if self.can_solve():
                    print("SOLVED")
                    return True
                    
            print("#i:{} loss: {}, mean reward: {}".format(i, loss, reward))
        return False    
            
            

    
    
    def optimize(self, obs, act, weights):
        '''Optimizing logits with loss, using self.train_op(optimizer)'''
        return self.sess.run([self.loss, self.train_op], feed_dict = {
            self.obs_ph : obs,
            self.act_ph: act,
            self.weights_ph: weights
        })

if __name__ == "__main__":
    agent = DiscreteRewardToGoAgent("InvertedPendulumPyBulletEnv-v0")#gym.make("InvertedPendulumPyBulletEnv-v0")
    agent.train_n_epochs(500)
    agent.play_episode(True)
    agent.env.close()
