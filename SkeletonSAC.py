import numpy as np
import tensorflow as tf
import time
import pickle
import operator
from osim.env import L2M2019Env

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)

# Helper functions
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

def neural_network(input_shape,activation,actor):
	if actor == True:
		model = tf.keras.Sequential([tf.keras.layers.Dense(256,activation = activation,input_shape = input_shape, dtype = 'float32'),
			#tf.keras.layers.Dense(256,activation = activation),
			tf.keras.layers.Dense(256,activation = activation)])
	else:
		model = tf.keras.Sequential([tf.keras.layers.Dense(256,activation = activation,input_shape = input_shape, dtype = 'float32'),
			#tf.keras.layers.Dense(256,activation = activation),
			tf.keras.layers.Dense(256,activation = activation),
			tf.keras.layers.Dense(1,activation = None)])
	return model

class Agent():
	def __init__(self):

		self.learning_rate = 0.001
		self.polyak = 0.005
		self.gamma = 0.99 
		self.alpha = 0.05
		#self.alpha = 0.05
		self.epochs = 10000
		self.max_episode_length = 10000


		self.state_desc = None
		self.prev_state_desc = None

		self.action_space = 22
		self.observation_space = 339

		# Buffer
		self.buffer = {'state':[],'rewards':[],"actions":[],"next_state":[],"done_flags":[]}
		self.buffer_length = 0
		self.max_buffer_size = 100000
		self.batch_size = 256


		# Actor parameters --> net, mean --> layer(net), log_sima --> layer(net)
		self.net = neural_network([self.observation_space],tf.nn.relu,True)
		self.mean = tf.keras.Sequential([tf.keras.layers.Dense(self.action_space,activation = tf.sigmoid, input_shape = [256], dtype = 'float32')])
		self.log_sigma = tf.keras.Sequential([tf.keras.layers.Dense(self.action_space, activation = tf.tanh, input_shape = [256], dtype = 'float32')])

		# Critic networks --> two Q func (TD3 trick), one value func
		self.Q_one = neural_network([self.observation_space + self.action_space], tf.nn.relu, False)
		self.Q_two = neural_network([self.observation_space + self.action_space], tf.nn.relu, False)
		self.V_func = neural_network([self.observation_space], tf.nn.relu, False)

		# Target functions
		self.target_V_func = tf.keras.models.clone_model(self.V_func)
		self.target_V_func.build()
		self.target_V_func.set_weights(self.V_func.get_weights())

		# Loading the networks
		#self.net = tf.keras.models.load_model('C:/Users/vlpap/Desktop/Skeleton_Weights/net.h5')
		#self.mean = tf.keras.models.load_model('C:/Users/vlpap/Desktop/Skeleton_Weights/mean.h5')
		#self.log_sigma = tf.keras.models.load_model('C:/Users/vlpap/Desktop/Skeleton_Weights/log_sigma.h5')
		#self.Q_one = tf.keras.models.load_model('C:/Users/vlpap/Desktop/Skeleton_Weights/Q_one.h5')
		#self.Q_two = tf.keras.models.load_model('C:/Users/vlpap/Desktop/Skeleton_Weights/Q_two.h5')
		#self.V_func = tf.keras.models.load_model('C:/Users/vlpap/Desktop/Skeleton_Weights/V_func.h5')
		#self.target_V_func = tf.keras.models.load_model('C:/Users/vlpap/Desktop/Skeleton_Weights/target_V_func.h5')



	def get_action(self, ob, k):

		net_val = self.net.predict(ob)
		mu = self.mean.predict(net_val)
		sig = np.exp(self.log_sigma.predict(net_val))
		ac = (mu + sig * np.random.normal(0,1,self.action_space)) 
		ac = np.clip(ac,0,1)[0]

		return ac

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

	def update(self,obs, rews, acs, obs_next, done_flags):

		# Q_func updates
		target = rews.reshape(-1,1) + self.gamma * (1-done_flags.reshape(-1,1)) * self.target_V_func.predict(obs_next)
		sa_pair = np.concatenate([obs,acs], axis = 1)
		param = []
		for var in self.Q_one.trainable_variables:
			param.append(var)
		for var in self.Q_two.trainable_variables:
			param.append(var)
		with tf.GradientTape() as tape:
			tape.watch(param)
			loss = tf.reduce_mean((self.Q_one(sa_pair) - target)**2) + tf.reduce_mean((self.Q_two(sa_pair) - target)**2)
			grad = tape.gradient(loss,param)
		optimizer = tf.keras.optimizers.Adam(self.learning_rate)
		optimizer.apply_gradients(zip(grad, param))
		#print("Q func loss", loss)


		# V_func update
		param = []
		for var in self.V_func.trainable_variables:
			param.append(var)
		with tf.GradientTape() as tape:

			net_val = self.net.predict(obs)
			mu = self.mean.predict(net_val)
			sig = np.exp(self.log_sigma.predict(net_val))
			acs_bar = (mu + sig * np.random.normal(0,1,sig.shape))
			acs_bar = np.clip(acs_bar,0,1)
			sa_pair = np.concatenate([obs, acs_bar], axis = 1)
			q_val = tf.minimum(self.Q_one.predict(sa_pair).reshape(-1,), self.Q_two.predict(sa_pair).reshape(-1,))
			log_pi = tf.reduce_sum(-0.5 * (((acs_bar - mu)/(sig+ 1e-06))**2 ), axis = 1)
			log_pi = tf.cast(log_pi, tf.float32)
			target = q_val - self.alpha * log_pi
			tape.watch(param)
			v = self.V_func(obs)
			v = tf.reshape(v, [-1,])
			loss = 0.5 * tf.reduce_mean((v - target)**2)
			grad = tape.gradient(loss, param)
		optimizer.apply_gradients(zip(grad, param))
		#print("V func loss", loss)

		

		# Policy update
		param = []
		for var in self.net.trainable_variables:
			param.append(var)
		for var in self.mean.trainable_variables:
			param.append(var)
		for var in self.log_sigma.trainable_variables:
			param.append(var)

		with tf.GradientTape() as tape:
			tape.watch(param)
			net_val = self.net(obs)
			mu = self.mean(net_val)
			sig =  tf.exp(self.log_sigma(net_val))
			#print("MEAN ",mu)
			acs_bar = (mu + sig * tf.random.normal(tf.shape(mu),0,1))
			acs_bar = tf.clip_by_value(acs_bar,0,1)
			#print("SIGMA ",sig)
			log_pi = tf.reduce_sum(-0.5*(((acs_bar-mu)/(sig+ 1e-06))**2), axis = 1)
			sa_pair = tf.concat([obs,acs_bar], axis = 1)
			log_pi = tf.reshape(log_pi,[-1,1])
			loss = tf.reduce_mean(self.alpha * log_pi - self.Q_one(sa_pair))
			grad = tape.gradient(loss, param)
		optimizer.apply_gradients(zip(grad, param))
		#print("Policy loss ",loss)



		# Target update
		for i in range(len(self.V_func.trainable_variables)):
			self.target_V_func.trainable_variables[i].assign(self.target_V_func.trainable_variables[i] * (1 - self.polyak)\
			 + self.polyak * self.V_func.trainable_variables[i])

		# End


	def final(self):
		env = L2M2019Env(difficulty = 1,visualize = False)
		for i in range(self.epochs):
			# Run environment
			print("####################### EPISODE NUMBER ", i," ##############################")
			t1 = time.time()
			for k in range(5):
				ob = env.reset()
				ob_desc = env.get_state_desc()
				for j in range(self.max_episode_length):
					self.prev_state_desc = ob_desc
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
					ac = self.get_action(ob, k)
					ob, rew, done, _ = env.step(ac)
					#if j % 10 == 0:
					#	print(ac)
					ob_desc = env.get_state_desc()
					self.state_desc = ob_desc
					rew = reward_calc(self.state_desc, self.prev_state_desc, ob_dict)
					#print(rew)
					self.buffer['next_state'].append(convert_to_array(ob))
					self.buffer['rewards'].append(rew)
					self.buffer['actions'].append(ac)
					if done:
						self.buffer['done_flags'].append(1)
					else:
						self.buffer['done_flags'].append(0)
					self.buffer_length += 1

					if done:
						break	

			for j in range(200):	

				# Gradient step
				batch = self.select_batch()
				obs, rews, acs, obs_next, done_flags = batch
				self.update(obs, rews, acs, obs_next, done_flags)		

			t2 = time.time()
			print("The episode finished in ", t2 - t1)



			if (i+1) % 5 == 0:
				ob = env.reset()
				total_reward = 0

				for k in range(self.max_episode_length):
					#env.render()
					ob_desc = env.get_state_desc()
					ob = convert_to_array(ob)
					ob = np.array(ob).reshape(1,-1)
					ac = self.mean.predict(self.net.predict(ob))  
					ac = ac[0]
					ob, rew, done, _ = env.step(ac)
					total_reward += rew
					if done:
						break
						
				print("The current size of the buffer is ",self.buffer_length)
				print(" The performance of the deterministic policy at ",i+1," is ",total_reward)
				self.net.save('C:/Users/vlpap/Desktop/Skeleton_Weights/net.h5')
				self.mean.save('C:/Users/vlpap/Desktop/Skeleton_Weights/mean.h5')
				self.log_sigma.save('C:/Users/vlpap/Desktop/Skeleton_Weights/log_sigma.h5')
				self.Q_one.save('C:/Users/vlpap/Desktop/Skeleton_Weights/Q_one.h5')
				self.Q_two.save('C:/Users/vlpap/Desktop/Skeleton_Weights/Q_two.h5')
				self.V_func.save('C:/Users/vlpap/Desktop/Skeleton_Weights/V_func.h5')
				self.target_V_func.save('C:/Users/vlpap/Desktop/Skeleton_Weights/target_V_func.h5')


agent = Agent()
agent.final()