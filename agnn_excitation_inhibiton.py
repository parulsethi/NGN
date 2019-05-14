# scale the input

# weight initialization
# normalization
# code normal neural network

import random
import numpy as np
import numpy.ma as ma
import csv
import matplotlib.pyplot as plt


# X = (hours sleeping, hours studying), y = score on test
# X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# Y = np.array(([92], [86], [89]), dtype=float)
Y = np.array([[0,0,1,1, 1, 0, 0]])

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
Y = Y/100 # max test score is 100


def read_input(filename, Y_pos, X_pos):
	data = csv.reader(open(filename, 'r'), delimiter=",")
	Y = [] # ground truth values
	X = [] # input data features 

	for row in data:
		Y.append(row[Y_pos])
		X.append(list(row[X_pos[0]:X_pos[1]]))

	X = np.array(X).astype(np.int)
	Y = np.array([Y]).astype(np.int)

	# scale units
	X = X/np.amax(X, axis=0) # maximum of X array
	Y = Y/4 # max score is 4, coverting to 0 or 1 due to classification tasks
	for i in range(len(Y[0])):
		if Y[0][i]==0.5:
			Y[0][i]=0
		else:
			Y[0][i]=1

	return Y, X


def NGN(X, weights, hidden_size, subset_size, output_size, neuron_glia_power):
	# Neuron Glia Network
	Y = []
	input_size = len(X[0])
	no_of_subset = int(hidden_size/subset_size)
	sensitivity = 2 # minimum no. of iterations to activate astrocytes
	neuron_counter = np.zeros(hidden_size)
	astrocyte_activation_counter = np.zeros(hidden_size)
	astrocyte_inhibition_counter = np.zeros(hidden_size)

	chromosome_w1 = input_size*hidden_size
	chromosome_wh = hidden_size*subset_size
	chromosome_w2 = hidden_size*output_size

	# weights
	W1 = weights[:chromosome_w1].reshape(input_size, hidden_size)
	WH = weights[chromosome_w1:chromosome_w1+chromosome_wh].reshape(subset_size, hidden_size)
	W2 = weights[chromosome_w1+chromosome_wh:].reshape(hidden_size, output_size)

	# dot product of X (input) and first set of weights W1
	hidden_summation = np.matmul(X, W1)

	for x in range(len(X)):
		for i in range(neuron_glia_power):
			# intralayer weight multiplication
			# multiply each column of WH by corresponding hidden_summation's value
			p = np.multiply(WH, hidden_summation[x])
			hidden_activation = []
			for i in range(no_of_subset):
				# summate subset rows
				y = np.sum(p[:, subset_size*i:(subset_size*i)+subset_size], axis=1)
				hidden_activation.extend(y)
			hidden_activation = sigmoid(np.array(hidden_activation)) # activation function

			# convert activation to binary value
			current_activation_thresholded = (hidden_activation>0.5).astype(int)
			# update neuron activity counter
			neuron_counter = neuron_counter + current_activation_thresholded

			# modifications according to time factor
			for i in range(len(neuron_counter)):
				# neuron counter satisfies time factor and activates astrocyte

				if astrocyte_inhibition_counter[i]==1:
					# increase the neuron's outgoing intralayer weights (increase the neuron's column values in WH) 
					WH[:,i]-=0.1
					astrocyte_inhibition_counter[i] = 0

				if astrocyte_activation_counter[i]==1:
					# decrease the neuron's outgoing intralayer weights (increase the neuron's column values in WH) 
					WH[:,i]-=0.1
					astrocyte_activation_counter[i] = 0
					astrocyte_inhibition_counter[i]+=1

				if neuron_counter[i]==sensitivity:
					# increase the neuron's outgoing intralayer weights (increase the neuron's column values in WH) 
					WH[:,i]+=0.25
					astrocyte_activation_counter[i]+=1
					# reset neuron counter
					neuron_counter[i] = 0

		# dot product of hidden layer (z2) and second set of 36x1 weights
		z3 = np.dot(hidden_activation, W2)
		# final activation function	
		o = sigmoid(z3)

		Y.append(o[0])

	return Y

def NN(X, weights, hidden_size, output_size):

	chromosome_w1 = len(X[0])*hidden_size
	chromosome_w2 = hidden_size*output_size

	# weights
	W1 = weights[:chromosome_w1].reshape(len(X[0]), hidden_size)
	W2 = weights[chromosome_w1:].reshape(hidden_size, output_size)

	# dot product of X (input) and W1
	z = np.matmul(X, W1)
	# activation function
	z2 = sigmoid(z)
	# dot product of hidden layer (z2) and W2
	z3 = np.matmul(z2, W2)
	# final activation function
	o = sigmoid(z3)
	
	return o
	
		

def cost_function(O, Y):
	# determines the mean square difference between the expected Y and our actual Y
	cost = np.square(Y - O).mean()
	return cost
	

def sigmoid(s):
	# activation function
	return 1/(1+np.exp(-s))


def Genetic_Evolve(Y, X, hidden_size, subset_size, output_size, neuron_glia_power, algo):

	init_population = 100
	mutation_rate = 0.05
	num_generations = 22
	winners_per_gen = 20
	error_reducing_trend = []

	if algo == 'NGN':
		chromosome = (len(X[0])*hidden_size)+(hidden_size*subset_size)+(hidden_size*output_size)
	else:
		chromosome = (len(X[0])*hidden_size)+(hidden_size*output_size)
	
	# initialize current population to random values within range
	current_population = np.random.uniform(0, 1, init_population*chromosome).reshape(init_population, chromosome)

	# initialize next population array
	next_population = np.zeros((current_population.shape[0], current_population.shape[1]))
	# 1st column is index of the individual in population, 2nd column is cost
	fit_vector = np.zeros((init_population, 2))

	# iterate through every generation
	for current_generation in range(num_generations):
		for x in range(init_population):
			if algo == 'NGN':
				O = NGN(X, current_population[x], hidden_size, subset_size, output_size, neuron_glia_power)
			else:
				O = NN(X, current_population[x], hidden_size, output_size)
			cost = np.sum(cost_function(O, Y))
			# create vec of all errors from cost function
			fit_vector[x] = np.array([x, cost])
		print("(Generation: #%s) Total error: %s\n" % (current_generation, np.sum(fit_vector[:,1])))
		error_reducing_trend.append([current_generation, np.sum(fit_vector[:,1])])
		
		winners = np.zeros((winners_per_gen, chromosome))

		for n in range(winners_per_gen):
			# select 20/2=10 index values randomly within range of fit_vector's length
			selected = np.random.choice(range(len(fit_vector)), int(winners_per_gen/2), replace=False)
			# find the index of minimum cost value from randomly selected individuals in fit_vector
			winner = np.argmin(fit_vector[selected, 1])
			winners[n] = current_population[int(fit_vector[selected[winner]][0])]

		# populate new generation with winners
		next_population[:len(winners)] = winners

		# populate rest of the generation with offspring of mating pairs
		next_population[len(winners):] = np.array([np.array(np.random.permutation(np.repeat(winners[:, i], ((init_population - len(winners))/len(winners)), axis=0))) for i in range(winners.shape[1])]).T

		# randomly mutate part of the population
		next_population = np.array(np.multiply(next_population, np.matrix([np.float(np.random.normal(0,2,1)) if random.random() < mutation_rate else 1 for x in range(next_population.size)]).reshape(next_population.shape)))

		current_population = next_population

	# return best solution weight
	return current_population[int(np.argmin(fit_vector[:, 1]))], error_reducing_trend

def plot(subset_error_values):

	values = np.array(subset_error_values).T
	plt.scatter(values[0], values[1])
	plt.xlabel("Subset Size")
	plt.ylabel("Error")
	plt.show()

Y, X = read_input(filename='breast-cancer-wisconsin.csv', Y_pos=10, X_pos=(1, 9))


output_size=1
neuron_glia_power=8
subset_sizes = [2, 3, 4, 6, 8, 9, 12, 18, 24, 36]
subset_error_values = []
error_trend_values = []

hidden_size = 72

for i in range(5):
	for subset_size in subset_sizes:
		solution_weight, error_reducing_trend = Genetic_Evolve(Y, X, hidden_size, subset_size, output_size, neuron_glia_power, 'NGN')
		soultion = NGN(X, solution_weight, hidden_size, subset_size, output_size, neuron_glia_power)
		error = cost_function(soultion, Y)
		subset_error_values.append([subset_size, error])
		error_trend_values.append([subset_size, error_reducing_trend])

with open('error_hs_excitation_inhibition_36.txt', 'w') as f:
	for item in subset_error_values:
		f.write("%s\n" % item)

with open('error_trend_hs_excitation_inhibition_36.txt', 'w') as f:
	for item in error_trend_values:
		f.write("%s\n" % item)


plot(subset_error_values)

# hidden_sizes = [36, 48, 120, 144, 168, 180, 210, 240, 252, 288, 300, 336, 408, 420, 450, 480, 630, 660, 720, 780, 840]
# hidden_size_error_values = []

# for i in range(5):
# 	for hidden_size in hidden_sizes:
# 		subset_size=0
# 		solution_weight = Genetic_Evolve(Y, X, hidden_size, subset_size, output_size, neuron_glia_power, 'NN')
# 		soultion = NN(X, solution_weight, hidden_size, output_size)
# 		error = cost_function(soultion, Y)
# 		hidden_size_error_values.append([hidden_size, error])

# plot(hidden_size_error_values)





