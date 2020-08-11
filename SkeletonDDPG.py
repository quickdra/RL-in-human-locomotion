import pandas as pd
import numpy as np
import gym
import multiprocessing
import pickle
import tensorflow as tf
import time
from osim.env import L2M2019Env
import operator
seed = 100
tf.random.set_seed(seed)
np.random.seed(seed)

#### Helper Functions
def neural_network(input_shape,activation,output_activation,output_shape):
	model = tf.keras.Sequential([tf.keras.layers.Dense(256,activation = activation,input_shape = input_shape),
		tf.keras.layers.Dense(256,activation = activation),
		#tf.keras.layers.Dense(64,activation = activation),
		tf.keras.layers.Dense(output_shape,activation = output_activation)])
	return model

def muscle(mus):
	arr = [mus['f'],mus['l'],mus['v']]
	return arr
	
def leg(leg_dict):
	leg = []
	ground = leg_dict['ground_reaction_forces']
	leg = leg + ground
	joint = leg_dict['joint']
	hip_abd = joint['hip_abd']
	hip = joint['hip']
	knee = joint['knee']
	ankle = joint['ankle']
	leg.append(hip_abd)
	leg.append(hip)
	leg.append(knee)
	leg.append(ankle)
	d_joint = leg_dict['d_joint']
	hip_abd = d_joint['hip_abd']
	hip = d_joint['hip']
	knee = d_joint['knee']
	ankle = d_joint['ankle']
	leg.append(hip_abd)
	leg.append(hip)
	leg.append(knee)
	leg.append(ankle)
	HAB = leg_dict['HAB']
	leg = leg + muscle(HAB)
	HAD = leg_dict['HAD']
	leg = leg + muscle(HAD)
	HFL = leg_dict['HFL']
	leg = leg + muscle(HFL)
	GLU = leg_dict['GLU']
	leg = leg + muscle(GLU)
	HAM = leg_dict['HAM']
	leg = leg + muscle(HAM)
	RF = leg_dict['RF']
	leg = leg + muscle(RF)
	VAS = leg_dict['VAS']
	leg = leg + muscle(VAS)
	BFSH = leg_dict['BFSH']
	leg = leg + muscle(BFSH)
	GAS = leg_dict['GAS']
	leg = leg + muscle(GAS)
	SOL = leg_dict['SOL']
	leg = leg + muscle(SOL)
	TA = leg_dict['TA']
	leg = leg + muscle(TA)
	return leg


def convert_to_array(observation):
	#V_TARGET
	v_tgt_field = observation['v_tgt_field'].reshape(1,242)
	v_tgt_field = v_tgt_field[0]

	#PELVIS
	pelvis = observation['pelvis']
	height = pelvis['height']
	pitch = pelvis['pitch']
	roll = pelvis['roll']
	vel = pelvis['vel']
	pelvis = []
	pelvis.append(height)
	pelvis.append(pitch)
	pelvis.append(roll)
	pelvis = pelvis + vel

	#RIGHT LEG
	r_leg = observation['r_leg']
	right_leg = leg(r_leg)

	#LEFT_LEG
	l_leg = observation['l_leg']
	left_leg = leg(l_leg)

	output = np.concatenate([v_tgt_field, np.array(pelvis + right_leg + left_leg)])
	output = output.tolist()
	return output


#### The agent starts here 
class Agent():
	def __init__(self):
		self.learning_rate = 0.0001
		self.max_episode_length = 10000
		self.gamma = 0.99
		self.polyak = 0.005
		self.epochs = 10000

		self.obs_space = 339
		self.acs_space = 22

		#### Building actor and critic
		self.Q_network = neural_network([self.obs_space + self.acs_space],tf.nn.relu,None,1)
		self.policy = neural_network([self.obs_space],tf.nn.relu,tf.nn.sigmoid,self.acs_space)
		self.policy = tf.keras.models.load_model('C:/Users/vlpap/Desktop/policy.h5')
		self.target_policy = tf.keras.models.load_model('C:/Users/vlpap/Desktop/target_policy.h5')
		self.Q_network = tf.keras.models.load_model('C:/Users/vlpap/Desktop/Q_network.h5')
		self.target_Q_network = tf.keras.models.load_model('C:/Users/vlpap/Desktop/target_Q_network.h5')

		#### Building target networks ######
		#self.target_Q_network = tf.keras.models.clone_model(self.Q_network)
		#self.target_Q_network.build()
		#self.target_Q_network.set_weights(self.Q_network.get_weights())
		#self.target_policy = tf.keras.models.clone_model(self.policy)
		#self.target_policy.build()
		#self.target_policy.set_weights(self.policy.get_weights())

		#### Initializing replay buffer
		self.buffer = {"state":[],"rewards":[],"actions":[], "next_state":[], "done_flags":[]}
		self.buffer_length = 0
		self.max_buffer_size = 100000
		self.batch_size = 128

	
	def get_exploration_action(self,ob,model,i):
		action = model.predict(ob)
		random = tf.random.normal(tf.shape(action)) 
		action = action + 0.1 * random
		action = np.array(tf.clip_by_value(action,0,1))
		return action

	##### This method selects batch as uncorrelated data points
	def select_batch(self):
		if self.buffer_length == 1:
			return (np.array(self.buffer['state']), np.array(self.buffer['rewards']), np.array(self.buffer['actions']),\
			np.array(self.buffer['next_state']), np.array(self.buffer['done_flags']))

		if self.buffer_length <= self.batch_size:
			index = np.random.randint(0,self.buffer_length,self.buffer_length)

		else:
			index = np.random.randint(0,self.buffer_length,self.batch_size)

		f = operator.itemgetter(*index)

		batch = (np.array(f(self.buffer['state'])), np.array(f(self.buffer['rewards'])), np.array(f(self.buffer['actions'])),\
			np.array(f(self.buffer['next_state'])), np.array(f(self.buffer['done_flags'])))
		return batch


	def calc_target_Q(self,rews,obs_next, done_flags):
		target = []
		l = len(obs_next)
		for i in range(l):
			action = self.target_policy.predict(obs_next[i].reshape(1,-1))
			sa_pair = np.concatenate([obs_next[i], action[0]]).reshape(1,-1)
			q_next = self.target_Q_network.predict(sa_pair)[0][0]
			tar = rews[i] + (1 - done_flags[i]) * self.gamma * q_next
			target.append(tar)

		return target


	def update(self,target,acs,obs):
		#### Critic Update
		critic_param = self.Q_network.trainable_variables
		with tf.GradientTape() as tape:
			tape.watch(critic_param)
			#obs = np.array(obs)
			#acs = np.array(acs)
			sa_pair = np.concatenate([obs,acs], axis = 1)
			target = np.array(target).reshape(-1,1)
			v_loss = tf.reduce_mean((target - self.Q_network(sa_pair))**2)
		grad = tape.gradient(v_loss, critic_param)
		optimizer = tf.keras.optimizers.Adam(0.001)
		optimizer.apply_gradients(zip(grad, critic_param))


		#### Policy Update
		actor_param = self.policy.trainable_variables
		with tf.GradientTape() as tape:
			tape.watch(actor_param)
			policy_action = self.policy(obs)
			sa_pair = tf.concat([obs, policy_action], axis = 1)
			pi_loss = -tf.reduce_mean(self.Q_network(sa_pair))
			grad = tape.gradient(pi_loss, actor_param)
		optimizer = tf.keras.optimizers.Adam(0.0001)
		optimizer.apply_gradients(zip(grad, actor_param))

		#print("Q Function loss ",v_loss)
		#print("Pi Loss ",pi_loss)



	def target_update(self):
		
		# Updating critic target network
		l = len(self.Q_network.trainable_variables)
		for i in range(l):
			self.target_Q_network.trainable_variables[i].assign(self.polyak * self.Q_network.trainable_variables[i] + (1 - self.polyak) * self.target_Q_network.trainable_variables[i])

		#Updating policy target network
		l = len(self.policy.trainable_variables)
		for i in range(l):
			self.target_policy.trainable_variables[i].assign(self.polyak * self.policy.trainable_variables[i] + (1 - self.polyak) * self.target_policy.trainable_variables[i])


	def final(self):
		#### Environment Initialization
		env = L2M2019Env(visualize = False,integrator_accuracy = 0.005)

		#### Run the environment N number of times
		#### Occasionally check the deterministic policy performance
		for i in range(self.epochs):
			ob = env.reset()
			print("######## EPISODE NUMBER ",i," ########")
			t1 = time.time()
			for j in range(self.max_episode_length):
				ob = convert_to_array(ob)
				if len(self.buffer['state']) == self.max_buffer_size:
					self.buffer['state'] = self.buffer['state'][1:]
					self.buffer['next_state'] = self.buffer['next_state'][1:]
					self.buffer['rewards'] = self.buffer['rewards'][1:]
					self.buffer['actions'] = self.buffer['actions'][1:]
					self.buffer['done_flags'] = self.buffer['done_flags'][1:]
					self.buffer_length = self.buffer_length - 1
					
				self.buffer['state'].append(ob)
				ob = np.array(ob).reshape(1,-1)
				ac = self.get_exploration_action(ob,self.policy,i)
				ac = ac[0]
				ob, rew, done, _ = env.step(ac)
				#print("THE ACTION WAS ", ac)
				self.buffer['next_state'].append(convert_to_array(ob))
				self.buffer['rewards'].append(rew)
				self.buffer['actions'].append(ac)
				if done:
					self.buffer['done_flags'].append(1)
				else:
					self.buffer['done_flags'].append(0)
				self.buffer_length += 1

				#### Select randomly a batch to train on
				batch = self.select_batch()
				obs, rews, acs, obs_next, done_flags = batch


				#### Calculate target yi to update critic network
				target = self.calc_target_Q(rews, obs_next, done_flags)

				#### Update after getting the targets
				self.update(target, acs, obs)

				#### Update the target networks
				self.target_update()
				if done:
					break
			t2 = time.time()
			print("TIME TAKEN FOR THIS EPISODE ", t2-t1)

			
			#### Ocassionally checking the performance of the deterministic policy
			
			
			if (i + 1) % 5 == 0:
				ob = env.reset()
				total_reward = 0
				for k in range(self.max_episode_length):
					#env.render()
					ob = convert_to_array(ob)
					ob = np.array(ob).reshape(1,-1)
					ac = self.policy.predict(ob)
					ac = ac[0]
					ob, rew, done, _ = env.step(ac)
					total_reward += rew
					if done:
						break
				print(" The performance of the deterministic policy at ",i+1," is ",total_reward)
				print("The current size of the buffer is ",self.buffer_length)
			if (i + 1) % 10 == 0 and i > 100:
				self.policy.save('C:/Users/vlpap/Desktop/policy.h5')
				self.target_policy.save('C:/Users/vlpap/Desktop/target_policy.h5')
				self.Q_network.save('C:/Users/vlpap/Desktop/Q_network.h5')
				self.target_Q_network.save('C:/Users/vlpap/Desktop/target_Q_network.h5')



agent = Agent()
agent.final()


