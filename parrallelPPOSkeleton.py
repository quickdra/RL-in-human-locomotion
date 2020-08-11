#50 iterations are over
import multiprocessing
import pandas as pd 
import numpy as np 
import tensorflow as tf
import time
from osim.env import L2M2019Env


seed = 25
tf.random.set_seed(seed)
np.random.seed(seed)

########### HELPER FUNCTIONS #############
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
	return output

def neural_network(input_shape,activation,output_activation,output_shape):
	model = tf.keras.Sequential([tf.keras.layers.Dense(256,activation = activation,input_shape = input_shape),
		tf.keras.layers.Dense(256,activation = activation),
		#tf.keras.layers.Dense(2048,activation = activation),
		#tf.keras.layers.Dense(128,activation = activation),
		#tf.keras.layers.Dense(500,activation = activation),
		tf.keras.layers.Dense(output_shape,activation = output_activation)])
	return model

class Agent():
	def __init__(self,obs_space,acs_space,learning_rate,max_path_length,min_timesteps):
		self.observation_space = obs_space 
		self.action_space = acs_space
		self.learning_rate = learning_rate
		self.gamma = 0.99
		self.lam = 0.97
		self.clip_ratio = 0.2
		self.beta = 0.001
		self.max_path_length = max_path_length
		self.min_timesteps = min_timesteps
		self.epochs = 1
		self.num_of_updates = 1
		self.actor_model = neural_network([obs_space],tf.tanh,None,acs_space)
		self.critic_model = neural_network([obs_space],tf.tanh,None,1)
		self.log_sigma_actor = tf.Variable(-0.5 * tf.ones([acs_space]),name = "sigma_actor",dtype = tf.float32,trainable = True)
		#self.actor_model = tf.keras.models.load_model('C:/Users/vlpap/Desktop/actor_model.h5')
		#self.critic_model = tf.keras.models.load_model('C:/Users/vlpap/Desktop/critic_model.h5')
		#self.old_actor_model = tf.keras.models.load_model('C:/Users/vlpap/Desktop/old_actor_model.h5')
		self.old_actor_model = tf.keras.models.clone_model(self.actor_model)
		self.old_actor_model.build()
		self.old_actor_model.set_weights(self.actor_model.get_weights())
		self.log_sigma_oldactor = tf.Variable(-0.5 * tf.ones([acs_space],dtype = tf.float32))
		self.log_sigma_oldactor.assign(self.log_sigma_actor)
		


##### Computes actions according to a gaussian distribution given by mean and variance #####
		
		# inputs = observation/observations
		# returns = list of list of actions ----> [[action],[action],....]
	def get_action(self,ob,model,sig):
		mu = model.predict(ob)
		sigma = tf.exp(sig)
		ac = mu + tf.random.normal(tf.shape(mu)) * sigma
		ac = tf.clip_by_value(ac,0,1).numpy()
		return ac


##### Computes the predicted value functions from the critic model #####

		# inputs = obs (observations of all trajectories ----> [[ob,ob,....],[ob,ob,....]....])
		# outputs = vfunc ------> [[[v],[v],....],[[v],[v],....],.....] (numpy array)

	def predict_value_func(self,obs_critic):
		vfunc = []
		for obs in obs_critic:
			obs = np.array(obs)
			v = self.critic_model.predict(obs)
			vfunc.append(v)
			#print(v)
		vfunc = np.array(vfunc)
		return vfunc


##### Computes the vfunc which will be later used as regression targets for the critic model ######

		# inputs = rews ([[rew,rew,...],[rew,rew,...],...])
		# outputs = vtarget ------> [[v_tar,v_tar,...],....]


	def compute_vtarget(self,rews):
		vtarget = []
		for i in range(len(rews)):
			r = rews[i]
			l = len(r)
			v = [0 for j in range(l)]
			v[-1] = r[-1]
			for j in range(l-2,-1,-1):
				v[j] = r[j] + self.gamma * v[j+1]
			vtarget.append(v)
		vtarget = np.array(vtarget)
		return vtarget



##### Calculates delta terms for every timestep in every trajectory #####

	# inputs = rewards, obs, value_function estimates (should be of the form [[rews of trajectory 1], [rews of trajectory 2], .......])
	# deltas = should be of the form [[deltas of trajectory 1], [deltas of trajectory 2], .....]

	def compute_delta(self,rews,vfunc):
		deltas = []
		for i in range(len(rews)):
			r = rews[i]
			v = vfunc[i]
			l = len(r)
			delta = []

			for j in range(l-1):
				v_t = v[j][0]
				v_t_next = v[j+1][0]
				d = r[j] + self.gamma * v_t_next - v_t
				delta.append(d)
			delta.append(r[-1] - v[-1][0])
			deltas.append(delta)
		return deltas

##### calculates general advantage estimates for every timestep in every trajectory ####

		# inputs = deltas
		# adv - general advantage estimates -----> [[ad,ad,...],[ad,ad,...],...]

	def compute_GAE(self,deltas):
		adv = []
		for i in range(len(deltas)):
			delta = deltas[i]
			a = [0 for x in range(len(delta))]
			a[-1] = delta[-1]
			for j in range(len(delta)-2,-1,-1):
				a[j] = delta[j] + a[j+1] * self.gamma * self.lam
			adv.append(a)
		return adv 



###### Updates the required parameters for the new actor and the critic models #######

		# inputs = acs, obs, rews
		# returns nothing

	def policy_update(self,acs,obs,adv,vtar,lr):
		# updating actor parameters
		actor_parameters = []
		for p in self.actor_model.trainable_variables:
			actor_parameters.append(p)
		actor_parameters.append(self.log_sigma_actor)
		with tf.GradientTape() as tape:
			tape.watch(actor_parameters)
			#pi_old = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = acs,logits = self.old_actor_model.predict(obs))
			#pi = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = acs,logits = self.actor_model(obs))
			pi_old = -0.5 * (((acs - self.old_actor_model.predict(obs))/(tf.exp(self.log_sigma_oldactor) + 1e-6))**2 + 2*self.log_sigma_oldactor + np.log(2*np.pi))
			pi_old = tf.reduce_sum(pi_old,axis = 1)
			pi = -0.5 * (((acs - self.actor_model(obs))/(tf.exp(self.log_sigma_actor) + 1e-6)) ** 2 + 2*self.log_sigma_actor + np.log(2*np.pi))
			pi = tf.reduce_sum(pi,axis = 1)
			entropy = tf.reduce_sum(-pi)
			ratio = tf.exp(pi - pi_old)
			adv = tf.convert_to_tensor(adv, dtype=tf.float32)
			clipped = tf.clip_by_value(ratio,1 - self.clip_ratio,1 + self.clip_ratio)
			pi_loss = -1 * tf.reduce_mean(tf.minimum(ratio * adv, clipped * adv)) - self.beta * entropy
		grad = tape.gradient(pi_loss, actor_parameters)
		optimizer = tf.keras.optimizers.Adam(lr)
		optimizer.apply_gradients(zip(grad, actor_parameters))
		print("ENTROPY ", entropy)
		print("POLICY LOSS ",pi_loss)

		approx_kl = tf.reduce_mean(pi_old-pi)


		# updating critic parameters
		with tf.GradientTape() as tape:
			tape.watch(self.critic_model.trainable_variables)
			vfunc = self.critic_model(obs)
			v_loss = tf.reduce_mean((vtar - vfunc) ** 2)
		grad = tape.gradient(v_loss,self.critic_model.trainable_variables)
		optimizer = tf.keras.optimizers.Adam(self.learning_rate)
		optimizer.apply_gradients(zip(grad, self.critic_model.trainable_variables))
		print("VALUE LOSS ",v_loss)

		return lr

	def refresh_weights(self):
		self.old_actor_model.set_weights(self.actor_model.get_weights())
		self.log_sigma_oldactor.assign(self.log_sigma_actor)


###### Responsible for calling all other functions and running everything in order ######

def worker_function(conn,seed):
	tf.random.set_seed(seed)
	env = L2M2019Env(difficulty = 2,visualize = False)
	model = neural_network([339],tf.tanh,None,22)
	sig = tf.Variable(-0.5 * tf.ones([22]),dtype = tf.float32,trainable = True)
	max_path_length = 10000
	min_timesteps = 10
	while True:
		cmd,var1,var2 = conn.recv()

		if cmd == 'reset_weights':
			model.set_weights(var1)
			sig.assign(var2)

		elif cmd == 'run':
			exp_steps = 0
			obs = []
			acs = []
			rews = []
			while exp_steps < min_timesteps:
				steps = 0
				obset = []
				rewset = []
				acset = []
				done = False
				ob = env.reset()
				ob = convert_to_array(ob)
				obset.append(ob.tolist())
				while True:
					ob = np.array(ob).reshape(1,-1)
					mu = model.predict(ob)
					sigma = tf.exp(sig)
					ac = mu + tf.random.normal(tf.shape(mu)) * sigma
					ac = tf.clip_by_value(ac,0,1).numpy()
					ac = ac[0]
					ob, rew, done, _ = env.step(ac)
					ob = convert_to_array(ob)
					obset.append(ob.tolist())
					rewset.append(rew)
					acset.append(ac)
					steps += 1
					if done or steps > max_path_length:
						break
				obs.append(obset[:-1])
				acs.append(acset)
				rews.append(rewset)
				exp_steps += steps
				#print(exp_steps)
			conn.send(True)

		elif cmd == 'send_data':
			conn.send([obs,rews,acs])

		elif cmd == 'close':
			conn.close()
			break


class Worker():
	def __init__(self,seed):

		self.conn1, conn2 = multiprocessing.Pipe()
		self.process = multiprocessing.Process(target = worker_function, args = (conn2,seed))
		self.process.start()



if __name__ == '__main__':
	agent = Agent(339,22,0.001,10000,5000)
	worker = []
	lr = 0.00005
	for i in range(40,40+2):
		worker.append(Worker(i))
	n = len(worker)
	for index in range(agent.epochs):
		print('############################ EPOCH ', str(index),' ##############################')
		agent.refresh_weights()
		t1 = time.time()
		paths = {'observations':[],'rewards':[],'actions':[]}
		done = [False for index in range(n)]
		for i in range(n):
			worker[i].conn1.send(('reset_weights',agent.old_actor_model.get_weights(),agent.log_sigma_oldactor))

		for i in range(n):
			worker[i].conn1.send(('run',0,0))

		for i in range(n):
			done[i] = worker[i].conn1.recv()

		for i in range(n):
			worker[i].conn1.send(('send_data',0,0))
			arr = worker[i].conn1.recv()
			paths['observations'] += arr[0]
			paths['rewards'] += arr[1]
			paths['actions'] += arr[2]


		t2 = time.time()
		print("TIME ",t2-t1)
		rews = paths['rewards']
		rewards = 0
		for r in rews:
			rewards += sum(r)
		rewards = rewards/len(rews)
		print("PERFORMANCE ",rewards)
		obs = paths['observations']
		acs = paths['actions']
		vfunc = agent.predict_value_func(obs)
		vtarget = agent.compute_vtarget(rews)
		adv = agent.compute_GAE(agent.compute_delta(rews,vfunc))
		x = []
		for a in adv:
			for b in range(len(a)):
				x.append(a[b])
		adv = np.array(x)
		adv = adv.reshape(adv.shape[0],)
		adv = (adv - adv.mean())/(adv.std() + 1e-6)
		x = []
		for v in vtarget:
			for b in range(len(v)):
				x.append(v[b])
		vtarget = np.array(x)
		vtarget = vtarget.reshape(-1,1) 
		divergence = []
		divmean = 0
		obs = np.concatenate([ob for ob in obs])
		acs = np.concatenate([ac for ac in acs])
		l = len(obs)
		print(obs.shape)
		print(acs.shape)
		print(vtarget.shape)
		print(adv.shape)
		if (index+1) % 20 == 0:
			lr = lr/1.25

		for j in range(agent.num_of_updates):
			print(j)
			divergence = []
			#if divmean > 0.015:
			#	break
			for k in range(l//2048 + 1):
				if k == l//2048:
					lr = agent.policy_update(acs[k*2048:],obs[k*2048:],adv[k*2048:],vtarget[k*2048:],lr)
				else:
					lr = agent.policy_update(acs[k*2048:(k+1)*2048],obs[k*2048:(k+1)*2048],adv[k*2048:(k+1)*2048],vtarget[k*2048:(k+1)*2048],lr)
				#divergence.append(kl)
			#divergence = np.array(divergence)
			#divmean = np.mean(divergence)
		if (index+1) % 5 == 0:
			print("********** SAVED ***********")
			agent.actor_model.save('C:/Users/vlpap/Desktop/actor_model.h5')
			agent.critic_model.save('C:/Users/vlpap/Desktop/critic_model.h5')

	for i in range(n):
		worker[i].conn1.send(('close',0,0))
		worker[i].process.terminate()