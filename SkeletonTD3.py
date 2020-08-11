import pandas as pd
import numpy as np
import gym
import multiprocessing
import pickle
import tensorflow as tf
import time
import operator
seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)


########### HELPER FUNCTIONS #############
def muscle(mus):
	arr = [mus['f'],mus['l'],mus['v']]
	#arr = [mus['v']]
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
	v_tgt_field = observation['v_tgt_field']
	v_tgt_field = np.array(v_tgt_field).reshape(1,242)
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
	return output.tolist()


def reward_calc(state_desc,prev_state_desc,ob_dict):
	
	reward = max(state_desc['body_pos']['tibia_l'][0], state_desc['body_pos']['tibia_r'][0])
	l = state_desc['body_pos']['tibia_l'][0] - prev_state_desc['body_pos']['tibia_l'][0]
	r = state_desc['body_pos']['tibia_r'][0] - prev_state_desc['body_pos']['tibia_r'][0]
	reward = 10 * max(l,r)

	head = state_desc['body_pos']['head'][1]
	prev_head = prev_state_desc['body_pos']['head'][1]
	head_pen = head - prev_head

	penalty = 5 * head_pen + 0.05 - 0.1 * (state_desc['body_pos']['toes_r'][0] + state_desc['body_pos']['toes_l'][0])
	#0.05 reward to be alive each second?
	#print(penalty)

	reward += penalty

	return reward

def neural_network(input_shape,activation,output_activation,output_shape):
	model = tf.keras.Sequential([tf.keras.layers.Dense(256,activation = activation,input_shape = input_shape),
		tf.keras.layers.Dense(256,activation = activation),
		#tf.keras.layers.Dense(64,activation = activation),
		tf.keras.layers.Dense(output_shape,activation = output_activation)])
	return model

#### The agent starts here 
class Agent():
	def __init__(self):
		self.learning_rate = 0.0001
		self.max_episode_length = 10000
		self.gamma = 0.99
		self.polyak = 0.005
		self.epochs = 100000

		self.obs_space = 339
		self.acs_space = 22

		#### Building actor and critic
		self.Q_network_one = neural_network([self.obs_space + self.acs_space],tf.nn.relu,None,1)
		self.Q_network_two = neural_network([self.obs_space + self.acs_space], tf.nn.relu, None, 1)
		self.policy = neural_network([self.obs_space],tf.nn.relu,tf.tanh,self.acs_space)
		#self.policy = tf.keras.models.load_model('C:/Users/vlpap/Desktop/policy.h5')

		#### Building target networks ######
		self.target_Q_network_one = tf.keras.models.clone_model(self.Q_network_one)
		self.target_Q_network_one.build()
		self.target_Q_network_one.set_weights(self.Q_network_one.get_weights())

		self.target_Q_network_two = tf.keras.models.clone_model(self.Q_network_two)
		self.target_Q_network_two.build()
		self.target_Q_network_two.set_weights(self.Q_network_two.get_weights())

		self.target_policy = tf.keras.models.clone_model(self.policy)
		self.target_policy.build()
		self.target_policy.set_weights(self.policy.get_weights())


		#### Initializing replay buffer
		self.buffer = {"state":[],"rewards":[],"actions":[], "next_state":[],'done_flags':[]}
		self.buffer_length = 0
		self.max_buffer_size = 1000000
		self.batch_size = 256

	
	def get_exploration_action(self,ob,model):
		action = model.predict(ob)
		random = tf.random.normal(tf.shape(action))
		action = action + 0.1 * random
		action = np.array(tf.clip_by_value(action, 0, 1))
		return action

	##### This method selects batch size in order, Next implement the function such that the batch pairs picked are completely random
	##### and uncorrelated!
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



	def calc_target_Q(self,rews,obs_next,done_flags):
		target = []
		l = len(obs_next)
		for i in range(l):
			action = self.target_policy.predict(obs_next[i].reshape(1,-1)) 
			action = tf.clip_by_value(action + tf.random.normal(tf.shape(action),0,0.2),0,1)
			sa_pair = np.concatenate([obs_next[i], action[0]]).reshape(1,-1)
			q_next = tf.minimum(self.target_Q_network_one.predict(sa_pair)[0][0], self.target_Q_network_two.predict(sa_pair)[0][0])
			tar = rews[i] + (1 - done_flags[i]) * self.gamma * q_next
			target.append(tar)

		return target


	def update(self,target,acs,obs):
		#### Critic Update
		critic_param = []
		for var in self.Q_network_one.trainable_variables:
			critic_param.append(var)
		for var in self.Q_network_two.trainable_variables:
			critic_param.append(var)

		with tf.GradientTape() as tape:

			tape.watch(critic_param)
			#obs = np.array(obs)
			#acs = np.array(acs)
			sa_pair = np.concatenate([obs,acs], axis = 1)
			target = np.array(target).reshape(-1,1)
			v_loss = tf.reduce_mean((target - self.Q_network_one(sa_pair))**2) + tf.reduce_mean((target - self.Q_network_two(sa_pair))**2)
		grad = tape.gradient(v_loss, critic_param)
		optimizer = tf.keras.optimizers.Adam(0.001)
		optimizer.apply_gradients(zip(grad,critic_param))


		#### Policy Update
		if self.buffer_length % 2 == 0:
			actor_param = self.policy.trainable_variables
			with tf.GradientTape() as tape:
				tape.watch(actor_param)
				policy_action = self.policy(obs) 
				sa_pair = tf.concat([obs, policy_action], axis = 1)
				pi_loss = -tf.reduce_mean(self.Q_network_one(sa_pair))
				grad = tape.gradient(pi_loss, actor_param)
			optimizer = tf.keras.optimizers.Adam(0.0001)
			optimizer.apply_gradients(zip(grad, actor_param))

		#print("Q Function loss ",v_loss)
		#print("Pi Loss ",pi_loss)



	def target_update(self):
		
		# Updating critic target network
		l = len(self.Q_network_one.trainable_variables)
		for i in range(l):
			self.target_Q_network_one.trainable_variables[i].assign(self.polyak * self.Q_network_one.trainable_variables[i] + (1 - self.polyak) * self.target_Q_network_one.trainable_variables[i])

		l = len(self.Q_network_two.trainable_variables)
		for i in range(l):
			self.target_Q_network_two.trainable_variables[i].assign(self.polyak * self.Q_network_two.trainable_variables[i] + (1 - self.polyak) * self.target_Q_network_two.trainable_variables[i])

		#Updating policy target network
		l = len(self.policy.trainable_variables)
		for i in range(l):
			self.target_policy.trainable_variables[i].assign(self.polyak * self.policy.trainable_variables[i] + (1 - self.polyak) * self.target_policy.trainable_variables[i])


	def final(self):
		#### Environment Initialization
		env = gym.make('Pendulum-v0')
		flag = 0
		#### Run the environment N number of times
		#### Occasionally check the deterministic policy performance
		for i in range(self.epochs):
			ob = env.reset()
			ob_desc = env.get_state_desc()
			print("######## EPISODE NUMBER ",i," ########")
			t1 = time.time()
			for j in range(self.max_episode_length):
				prev = ob_desc
				ob_dict = ob
				ob = convert_to_array(ob)
				self.buffer['state'].append(ob)
				if len(self.buffer['state']) == self.max_buffer_size:
					self.buffer['state'] = self.buffer['state'][300:]
					self.buffer['next_state'] = self.buffer['next_state'][300:]
					self.buffer['rewards'] = self.buffer['rewards'][300:]
					self.buffer['actions'] = self.buffer['actions'][300:]
					self.buffer['done_flags'] = self.buffer['done_flags'][300:]
					self.buffer_length = self.buffer_length - 300
				ob = np.array(ob).reshape(1,-1)
				ac = self.get_exploration_action(ob,self.policy)
				ac = ac[0]
				ob, rew, done, _ = env.step(ac)
				#print("THE ACTION WAS ", ac)
				ob_desc = env.get_state_desc()
				curr = ob_desc
				rew = reward_calc(curr, prev, ob_dict)
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
				if self.buffer_length % 2 == 0:
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
					ob = np.array(ob).reshape(1,-1)
					ac = self.policy.predict(ob) 
					np.clip(ac,0,1)
					ac = ac[0]
					ob, rew, done, _ = env.step(ac)
					total_reward += rew
					if done:
						break
						
				print("The current size of the buffer is ",self.buffer_length)
				print(" The performance of the deterministic policy at ",i+1," is ",total_reward)
				self.policy.save('C:/Users/vlpap/Desktop/RL_solutions/policy.h5')
				self.target_policy.save('C:/Users/vlpap/Desktop/RL_solutions/target_policy.h5')
				self.Q_network_one.save('C:/Users/vlpap/Desktop/RL_solutions/Q_network_one.h5')
				self.target_Q_network_two.save('C:/Users/vlpap/Desktop/RL_solutions/target_Q_network_one.h5')
				self.Q_network_two.save('C:/Users/vlpap/Desktop/RL_solutions/Q_network_two.h5')
				self.target_Q_network_two.save('C:/Users/vlpap/Desktop/RL_solutions/target_Q_network_two.h5')



agent = Agent()
agent.final()


