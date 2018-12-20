
"""

	Battle Ship 

"""

# internal
import random

# external
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_ships( ship_sizes ):

	# gen hashes
	ship_hashes = np.linspace(11, 11*len(ship_sizes), len(ship_sizes))
	
	# create ships
	ships = []

	for i,s,h in zip( range(len(ship_sizes)), ship_sizes, ship_hashes ):

		ship_i = {
			'size' : s,
			'index' : i,
			'hash' : h
		}

		ships.append( ship_i )

	# return ships
	return ships

def empty_board( ships, board_dims ):
	""" create and return zerod numpy array representing game board. """

	board = np.zeros( shape=(len(ships), *board_dims) )

	return board

def rand_placement(ship, board_slice):

	#
	# generate possible coordinate arrays
	#

	x, y = np.arange(0, board_slice.shape[0]-1), np.arange(0, board_slice.shape[1]-1) 

	#
	# choose random orientation
	#

	rand_orientation = lambda : np.random.choice(['horizontal','vertical'])

	if rand_orientation() == 'vertical':

		# verticle orientation
		rand_y = np.random.choice(y)
		selected_x_segment = x 
		selected_y_segment = np.repeat(rand_y, y.shape) # constant

	else:

		# horizontal orientation
		rand_x = np.random.choice(x)
		selected_x_segment = np.repeat(rand_x, x.shape) # constant
		selected_y_segment = y 

	#
	# choose random segment
	#

	# first choose random endpoint indices
	cap = -ship['size'] #+ 1
	assert ship['size'] > 1
	x_start, y_start = np.random.choice(x[:cap]), np.random.choice(y[:cap])
	x_end, y_end = x_start + ship['size'], y_start + ship['size']

	# grab segment coordinates
	x_coors, y_coors = selected_x_segment[x_start:x_end], selected_y_segment[y_start:y_end]

	#
	# write ship (ones) to board slice in selected location
	#

	values = np.ones(shape=(ship['size'],))
	board_slice[x_coors, y_coors] = values

	#
	# probabilities screw to left side...
	#

	# randomly take the transpose (***new)
	if random.random() < 0.5:
		board_slice = board_slice.T

	# randomly flip board (***new) 
	if random.random() < 0.5:
		board_slice = np.flip(board_slice, axis=0)
	if random.random() < 0.5:
		board_slice = np.flip(board_slice, axis=1)

	# return board slice
	return board_slice

def random_board(ships, board_dims):

	# create empty board
	board = empty_board(ships, board_dims)

	# place ships randomly
	for i, ship in enumerate(ships):
		board[i] = rand_placement( ship, board[i] )

	# recurse until there are none overlapping
	if np.max(np.add.reduce(board, 0)) > 1:
		return random_board(ships, board_dims)

	return board

def display_board( ships, board ):

	board_copy = np.array(board, copy=True)

	#
	# replace board slice ones with corresponding ship hashes
	#

	for ship in ships:
		board_slice = board_copy[ship['index']]

		board_slice[board_slice > 0] = ship['hash']


	#
	# flatten and display
	#
	
	print(np.add.reduce(board_copy, 0))


def random_fire_pattern( board_dims, number_of_shots ):
	""" Generate random static fire/overlay pattern. """

	n_places = board_dims[0]*board_dims[1]

	# create flat board overlay
	flat = np.zeros(shape=(n_places,))

	# fill randomly with 1's (equal to number_of_shots)
	rand_indices = random.sample( range(n_places), number_of_shots )
	values = np.ones(shape=(number_of_shots,))
	np.put(flat, rand_indices, values)

	# reshape and return overlay fire pattern
	overlay = flat.reshape(board_dims)

	return overlay


def ships_hit_by_fire_pattern( board, fire_pattern ):
	""" find the number of unique ships hit by fire pattern, 
		and return this number. """

	ship_hit = lambda board_slice: 1 if np.max(np.add(board_slice, fire_pattern)) > 1 else 0

	ships_hit = 0
	for board_slice in board[:]:
		ships_hit += ship_hit(board_slice)

	return ships_hit


def fire_pattern_stats( ships, board_dims, fire_pattern, num_trials ):
	""" run trials on fire pattern and return mean and std of number of distinct ships hit """

	trial_hits = []

	for i in range(num_trials):

		# create random board
		board = random_board(ships, board_dims)
		#display_board(ships, board)

		# get number of distinct hits
		ship_hits = ships_hit_by_fire_pattern(board, fire_pattern)

		# update history
		trial_hits.append(ship_hits)

	trial_hits = np.array(trial_hits)

	# calculate stats
	mean_hits = np.mean(trial_hits)
	std_hits = np.std(trial_hits)

	# return stats
	return mean_hits, std_hits


def fire_pattern_search( ships, board_dims, max_its=100, num_trials=24, keep_top=5 ):

	top_fire_patterns = np.zeros(shape=(keep_top, *board_dims))
	top_fire_pattern_means = np.zeros(shape=(keep_top,))
	top_fire_pattern_stds = np.zeros(shape=(keep_top,))

	for i in tqdm(range(max_its)):

		# generate random fire pattern
		fire_pattern = random_fire_pattern(BOARD_DIMS, NUM_SHOTS)

		# run trials
		mean, std = fire_pattern_stats(ships, BOARD_DIMS, fire_pattern, num_trials=24)

		# update tops
		if np.min(top_fire_pattern_means) <= mean:

			if np.min(top_fire_pattern_stds) <= std:

				# get lowest index
				idx = top_fire_pattern_means.argmin()

				if top_fire_pattern_stds[idx] < std:

					# replace with new fire pattern
					top_fire_patterns[idx] = fire_pattern
					top_fire_pattern_means[idx] = mean
					top_fire_pattern_stds[idx] = std

	#
	# Sort Ascending Mean
	#

	sorted_indcies = np.argsort(top_fire_pattern_means)
	top_fire_patterns = top_fire_patterns[sorted_indcies]
	top_fire_pattern_means = top_fire_pattern_means[sorted_indcies]
	top_fire_pattern_stds = top_fire_pattern_stds[sorted_indcies]

	#
	# Display Top Results
	#

	print("\n\n TOP PATTERNS:")

	for fire_pattern, mean, std in zip(top_fire_patterns,top_fire_pattern_means,top_fire_pattern_stds):
		
		print("\n\nPattern:\n")
		print(fire_pattern)

		print("\nMean:\n")
		print(mean)

		print("\nSTD:\n")
		print(std)


	#
	# Display Non weighted Heat Map
	#

	heat_map = np.add.reduce(top_fire_patterns, 0)

	print("\n\nHEAT MAP:\n")
	print(heat_map)

	plt.imshow(heat_map, cmap='hot')
	plt.show()

	#
	# Display Combined Pattern Map with Weights based on rank
	#

	weights = np.logspace(2,5, keep_top)
	weighted_patterns = top_fire_patterns * weights[:,np.newaxis,np.newaxis]
	weighted_heat_map = np.add.reduce(weighted_patterns, 0)
	weighted_heat_map *= 100/weighted_heat_map.max()

	print("\n\nWEIGHTED HEAT MAP:\n")
	print(weighted_heat_map)

	plt.imshow(weighted_heat_map, cmap='hot', interpolation='nearest')
	plt.show()

	#
	# Display Top Pattern
	#

	print("\n\nTOP PATTERN:\n")
	print(top_fire_patterns[-1])

	plt.imshow(top_fire_patterns[-1], cmap='hot', alpha=0.8)
	plt.show()

if __name__ == "__main__":

	#
	# Testing
	#

	#
	# Settings
	#

	# ship sizes
	SHIP_SIZES = [2,3,3,4,5]

	# game board dimensions
	BOARD_DIMS = (10,10)

	# fire pattern size
	NUM_SHOTS = 4

	# fire pattern search space
	NUM_FIRE_PATTERNS = 2500

	# number of trials per pattern
	NUM_TRIALS = 75

	# number of top patterns to keep
	TOP = 25

	#
	# Create Ships
	#

	ships = create_ships(ship_sizes=SHIP_SIZES)

	#
	# Generate Fire Pattern Search
	#

	fire_pattern_search( ships, BOARD_DIMS, max_its=NUM_FIRE_PATTERNS, num_trials=NUM_TRIALS, keep_top=TOP  )

