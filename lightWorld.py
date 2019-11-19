import numpy as np
import random
from collections import defaultdict
import copy
from matplotlib import pyplot as plt
font = {'size' : '15'}
plt.rc('font', **font)

NUMACTIONS = 6

class lightWorld:
	def __init__(self, width = 5, height = 5):
		self.newLightWorld(width,height)
		self.initParams = copy.deepcopy((self.world,self.key, self.lock, self.door, self.agent, self.haveKey, self.doorOpen))

	def newLightWorld(self, width, height):
		self.width = width
		self.height = height
		self.world = self.blankWorld(width,height)
		#key
		self.key = self.insertItem(3)
		#unlock_point
		self.lock = self.insertItem(4, True)
		#doorway
		self.door = self.insertItem(5, True)
		#start point (keep static?)
		self.agent = self.insertItem(2)
		#for now no obstacles
		self.haveKey = False
		self.doorOpen = False

	def resetWorld(self):
		self.world, self.key, self.lock, self.door, self.agent, self.haveKey, self.doorOpen = copy.deepcopy(self.initParams)
		# print self.world

	def blankWorld(self, width, height):
		world = np.zeros((width+1, height+1))

		#outer walls
		for i in range(height+1):
			world[i][0] = 1
			world[i][width] = 1
			if i == 0 or i == height:
				for j in range(width+1):
						world[i][j] = 1
		return world

	def printWorld(self):
		for line in self.world:
			textLine = []
			for space in line:
				if space == 0:
					textLine.append(' ')
				elif space == 1:
					textLine.append('W')
				elif space == 2:
					textLine.append('S')
				elif space == 3:
					textLine.append('K')
				elif space == 4:
					textLine.append('L')
				elif space == 5:
					textLine.append('D')
				elif space == 6:
					textLine.append('G')
				elif space == 7:
					textLine.append('|')
			print(textLine)

	def insertItem(self, item, edge = False):
		r = 0
		c = 0
		blocked = True
		while blocked:
			if edge:
				numSpots = (self.width-1)*2 + (self.height-1)*2
				loc = random.randrange(0,numSpots)
				if loc < ((self.width-1) * 2):
					r = (loc % (self.width-1)) + 1
					y = (loc / (self.width-1)) * (self.height)
				else:
					loc -= ((self.width-1) * 2)
					r = (loc / (self.height-1)) * (self.width)
					c = (loc % (self.height-1)) + 1
				if self.world[r][c] == 1:
					blocked = False
			else:
				r = random.randrange(1,self.height)
				c = random.randrange(1,self.width)
				if self.world[r][c] == 0:
					blocked = False
		self.world[r][c] = item
		return (r,c)

	def takeAction(self, a):
		r,c = self.agent
		#move
		if a < 4:
			return self.move(r,c,a)
		#pickup
		elif a == 4:
			if self.agent == self.key:
				self.world[r][c] = 7
				self.key = (-1,-1)
				self.haveKey = True
				return 1
			else:
				return -1
		#press
		elif a == 5:
			if self.world[r][c+1] == 4 or self.world[r][c-1] == 4 or self.world[r-1][c] == 4 or self.world[r+1][c] == 4:
				if self.haveKey:
					self.world[self.door[0]][self.door[1]] = 6
					self.haveKey = False
					self.doorOpen = True
					return 1
				else:
					return -1
			else:
				return -1
		else:
			print "Unknown action: ", a
			return 0

	def move(self, r, c, a):
		destR = r
		destC = c
		if a == 0:
			destC += 1
		elif a == 1:
			destR += 1
		elif a == 2:
			destC -= 1
		else:
			destR -= 1
		dest = self.world[destR][destC]

		#move to open space, -1 reward
		if dest == 0:
			if self.agent == self.key:
				self.world[r][c] = 3
			else:
				self.world[r][c] = 0
			self.agent = (destR,destC)
			self.world[destR][destC] = 7
			return -1
		elif dest == 3:
			self.world[r][c] = 0
			self.agent = (destR, destC)
			return -1
		#reach open door
		elif dest == 6:
			self.world[r][c] = 0
			self.agent = (destR, destC)
			self.world[destR][destC] = 5
			return 1000
		#hit obstacle, don't move, -1 reward
		else:
			return -1

	#open doors emit red light, keys on floor emit green, locks emit blue
	def observe(self):
		r = self.agent[0]
		c = self.agent[1]

		#TODO: based on last action can compute this incrementally instead of scratch each time
		#12 sensors and coords of agent
		featureVec = np.zeros(16)
		# featureVec = np.zeros(22)
		featureVec[0] = r
		featureVec[1] = c
		featureVec[14] = self.haveKey
		featureVec[15] = self.doorOpen

		# featureVec[16] = self.key[0]
		# featureVec[17] = self.key[1]
		# featureVec[18] = self.lock[0]
		# featureVec[19] = self.lock[1]
		# featureVec[20] = self.door[0]
		# featureVec[21] = self.door[1]

		for i in range(20):
			lCoord = c - i
			uCoord = r - i
			rCoord = c + i
			dCoord = r + i
			lightVal = 1 - i*0.05
			#left
			if lCoord >= 0:
				lObj = self.world[r][lCoord]
				if lObj == 6:
					featureVec[2] = lightVal
				elif lObj == 3:
					featureVec[3] = lightVal
				elif lObj == 4:
					featureVec[4] = lightVal
			#up
			if uCoord >= 0:
				uObj = self.world[uCoord][c]
				if uObj == 6:
					featureVec[5] = lightVal
				elif uObj == 3:
					featureVec[6] = lightVal
				elif uObj == 4:
					featureVec[7] = lightVal
			#right
			if rCoord <= self.width:
				rObj = self.world[r][rCoord]
				if rObj == 6:
					featureVec[8] = lightVal
				elif rObj == 3:
					featureVec[9] = lightVal
				elif rObj == 4:
					featureVec[10] = lightVal
			#down
			if dCoord <= self.height:
				dObj = self.world[dCoord][c]
				if dObj == 6:
					featureVec[11] = lightVal
				elif dObj == 3:
					featureVec[12] = lightVal
				elif dObj == 4:
					featureVec[13] = lightVal
		return tuple(featureVec)

def argmax(vals):
	return np.random.choice(np.flatnonzero(vals == vals.max()))

def epsGreedy(Q,state, eps = 0.1):
	#split eps proba equally between all actions
	actionProbas = np.ones(NUMACTIONS, dtype = float) * eps / NUMACTIONS
	actionProbas[argmax(Q[state])] += (1.0 - eps)
	action = np.random.choice(np.arange(len(actionProbas)), p = actionProbas)
	return action

def printPolicy(Qvals, width, height):
	labels = ['R','D','L','U','K','P']
	policy = [['_']*(width+1) for i in range(height+1)]
	for r,c in Qvals.keys():
		policy[r][c] = labels[argmax(Qvals[(r,c)])]
	for row in policy:
		print row

def qLearning(world, numIters, discount_factor = 1.0, alpha = 1, epsilon = 0.1):
	Qvals = defaultdict(lambda: np.zeros(NUMACTIONS))
	episodeLens = np.zeros(numIters)
	episodeRewards = np.zeros(numIters)
	for i in range(numIters):
		world.resetWorld()
		# world = lightWorld(world.width,world.height)
		goalState = False
		state = world.observe()
		while not goalState:
			action = epsGreedy(Qvals, state)
			reward = world.takeAction(action)
			if reward == 1000:
				goalState = True
			episodeRewards[i] += reward
			if reward != 0:
				episodeLens[i] += 1

			sPrime = world.observe()
			aPrime = argmax(Qvals[sPrime])
			TDerr = reward + discount_factor*Qvals[sPrime][aPrime] - Qvals[state][action]
			Qvals[state][action] += alpha*TDerr

			# g = raw_input("next step")
			# world.printWorld()
			# print state, action

			state = sPrime
		if i % 100 == 0:
			print i, episodeLens[i]
	return Qvals, episodeLens, episodeRewards




#actions are:
# 0 - right
# 1 - down
# 2 - left
# 3 - up
# 4 - pickup
# 5 - press
if __name__ == '__main__':
	# #manual sim
	# BUG: walking over key w/o picking up overwrites visual of it
	# controlMappings = {'d' : 0, 's' : 1, 'a' : 2, 'w' : 3, 'k' : 4, 'p' : 5}
	# world = lightWorld(10,10)
	# world.printWorld()
	# netReward = 0
	# goal = False
	# actionsTaken = 0
	# for i in range(2):
	# 	while not goal:
	# 		print(world.observe())
	# 		action = controlMappings[raw_input("Select an action : ")]
	# 		reward = world.takeAction(action)
	# 		if reward != 0:
	# 			actionsTaken += 1
	# 			if reward == 1000:
	# 				goal = True
	# 		netReward += reward
	# 		world.printWorld()
	# 		print "agent pos: ", world.agent
	# 		print "reward: ", netReward
	# 	print "actions taken: ", actionsTaken
	# 	world.resetWorld()
	# 	goal = False

	#issue when agent doesn't sense any light, literally no info to inform it what to do
	allLens = []
	allRewards = []
	numEpochs = 100
	print "total iters: ", numEpochs 
	for i in range(numEpochs):
		print "iter ", i
		world = lightWorld(10,10)
		# world.printWorld()
		numIters = 200
		Qvals, epLens, epRewards = qLearning(world, numIters)
		allLens.append(epLens)
		allRewards.append(epRewards)
	plt.plot(range(numIters), np.mean(allLens,axis=0))
	plt.xlabel("Episode")
	plt.ylabel("Length")
	plt.show()
	plt.plot(range(numIters), np.mean(allRewards,axis=0))
	plt.xlabel("Episode")
	plt.ylabel("Reward")
	plt.show()
