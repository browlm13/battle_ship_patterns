
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

def board_stats( board, board_dims, num_trials, num_shots ):
	""" run trials on board arrangment and return mean and std of number of distinct ships hit """

	trial_hits = []

	for i in range(num_trials):

		# generate random fire pattern
		fire_pattern = random_fire_pattern(board_dims, num_shots)

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

from matplotlib import colors
def display_fire_pattern( fire_pattern, board_dims ):

	data = fire_pattern*100

	# create discrete colormap
	cmap = colors.ListedColormap(['blue', 'red'])

	bounds = [0, 1, 100]
	norm = colors.BoundaryNorm(bounds, cmap.N)

	fig, ax = plt.subplots()
	ax.imshow(data, cmap=cmap) #, norm=norm)

	# draw gridlines
	ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

	ax.set_xticks(np.arange(-0.5, board_dims[0], 1));
	ax.set_yticks(np.arange(-0.5, board_dims[1], 1));

	plt.show()


from scipy import stats
def fire_pattern_minimum_requirement( ships, board_dims, target_mean, confidence=0.95, max_its=1000, num_trials=24, num_shots=20 ):
	""" Find a pattern that has the target mean with provided confidence interval, 
		return first fire pattern to meet this requirement. """


	z = stats.norm.ppf(confidence)

	# max
	max_fire_pattern = random_fire_pattern(board_dims, num_shots)
	max_mean = 0.0
	max_std = 0.0
	max_diff = np.inf
	max_range = 0.0
	max_slack = 0.0

	for i in tqdm(range(max_its)):

		# generate random fire pattern
		fire_pattern = random_fire_pattern(board_dims, num_shots)

		# run trials
		mean, std = fire_pattern_stats(ships, board_dims, fire_pattern, num_trials=num_trials)

		
		# check if criteria is met
		#scipy.stats.norm(mean=mean, std=std)
		interval_size = abs(z*std/np.sqrt(num_trials)) ## margin of error
		diff = target_mean - mean
		trial_range = mean + abs(interval_size)
		trial_slack = mean - abs(interval_size)

		#if max_diff > diff:
		if max_range <= trial_range:
			max_fire_pattern = fire_pattern
			max_mean = mean
			max_std = std
			max_diff = diff
			max_range = trial_range
			max_slack = trial_slack

			print("\n\n %s +- %s" % (max_mean, interval_size))
			print("confidence: ", confidence)
			print("std: ", max_std)
			print("diff: ", max_diff)
			print("slack: ", max_slack)
			print("stretch: ", max_range)

		if diff <= interval_size:

			print("\n\nFOUND.")

			print("\n\n %s +- %s" % (max_mean, interval_size))
			print("confidence: ", confidence)
			print("std: ", max_std)
			print("diff: ", max_diff)
			print("slack: ", max_slack)
			print("stretch: ", max_range)

			#
			# Display Top Pattern
			#

			print("\n\nPATTERN FOUND:\n")
			print(max_fire_pattern)

			plt.imshow(max_fire_pattern, cmap='hot', alpha=0.8)
			plt.show()


			return max_fire_pattern, max_mean, max_std

	print("\nFailure to find pattern given criteria")
	return max_fire_pattern, max_mean, max_std

def ship_arrangment_minimum_requirment(ships, board_dims, target_mean, confidence=0.95, max_its=1000, num_trials=24, num_shots=20 ):

	z = stats.norm.ppf(confidence)

	# min
	min_board = random_board(ships, board_dims)
	min_mean = np.inf
	min_std = np.inf
	min_diff = np.inf 
	min_range = np.inf
	min_slack = np.inf

	for i in tqdm(range(max_its)):

		# generate random board
		board = random_board(ships, board_dims)

		# run trials
		mean, std = board_stats(board, board_dims, num_trials, num_shots)

		# check if criteria is met
		interval_size = abs(z*std/np.sqrt(num_trials)) ## margin of error
		diff = mean - target_mean
		trial_range = mean + abs(interval_size)
		trial_slack = mean - abs(interval_size)

		#if max_diff > diff:
		if max_range <= trial_range:
			min_board = board
			min_mean = mean
			min_std = std
			min_diff = diff
			min_range = trial_range
			min_slack = trial_slack

			print("\n\n %s +- %s" % (min_mean, interval_size))
			print("confidence: ", confidence)
			print("std: ", min_std)
			print("diff: ", min_diff)
			print("slack: ", min_slack)
			print("stretch: ", min_range)

		if diff <= interval_size:

			print("\n\nFOUND.")

			print("\n\n %s +- %s" % (min_mean, interval_size))
			print("confidence: ", confidence)
			print("std: ", min_std)
			print("diff: ", min_diff)
			print("slack: ", min_slack)
			print("stretch: ", min_range)

			#
			# Display Top Board Arrangment
			#

			print("\n\BOARD FOUND:\n")
			display_board(min_board)

			return min_board, min_mean, min_std

	print("\nFailure to find board given criteria")
	return min_board, min_mean, min_std

def fire_pattern_search( ships, board_dims, max_its=100, num_trials=24, keep_top=5, num_shots=20 ):

	top_fire_patterns = np.zeros(shape=(keep_top, *board_dims))
	top_fire_pattern_means = np.zeros(shape=(keep_top,))
	top_fire_pattern_stds = np.zeros(shape=(keep_top,))

	for i in tqdm(range(max_its)):

		# generate random fire pattern
		fire_pattern = random_fire_pattern(board_dims, num_shots)

		# run trials
		mean, std = fire_pattern_stats(ships, board_dims, fire_pattern, num_trials=num_trials)

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
	NUM_SHOTS = 20

	# fire pattern search space
	NUM_FIRE_PATTERNS = 1000

	# number of trials per pattern
	NUM_TRIALS = 175

	# number of top patterns to keep
	TOP = 45

	#
	# Create Ships
	#

	ships = create_ships(ship_sizes=SHIP_SIZES)

	#
	# find fire pattern by criteria
	#
	TARGET_MEAN = len(SHIP_SIZES)
	CONFIDENCE = 0.975
	MAX_ITS = 100000
	NUM_SHOTS = 24
	SAMPLE_SIZE = 250

	fire_pattern, mean, std = fire_pattern_minimum_requirement( ships, BOARD_DIMS, TARGET_MEAN, confidence=CONFIDENCE, max_its=MAX_ITS, num_trials=SAMPLE_SIZE, num_shots=NUM_SHOTS )

	print("\n%s shots to hit %s distinct enemy ships." % (NUM_SHOTS, mean))
	
	print("\n confidence: ", CONFIDENCE)
	print("\n mean: ", mean)
	print("\n std: ", std)

	print("\n\nPATTERN:\n")
	
	print(fire_pattern)
	display_fire_pattern(fire_pattern, BOARD_DIMS)

	#
	# Generate Fire Pattern Search
	#

	fire_pattern_search( ships, BOARD_DIMS, max_its=NUM_FIRE_PATTERNS, num_trials=NUM_TRIALS, keep_top=TOP, num_shots=NUM_SHOTS  )


