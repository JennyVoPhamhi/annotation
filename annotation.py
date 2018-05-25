 # Function for importing and parsing Quantius results
import json
import pandas as pd

import numpy as np
from numpy import genfromtxt

import scipy

# Modules for plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Modules for clustering
import sklearn as skl
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.neighbors import KDTree

# ------- #

class Annotation:

	""" constructor takes in json filename and builds dataframe """
	def __init__(self, filename):
		
		self.annotations = pd.DataFrame()

		json_string = open(filename).read()
		results = json.loads(json_string)

		for turker in results:

			# Skip the turker if they didn't perform any annotations
			if not turker['raw_data']:
				continue

			# Make a data frame of the coordinates of each annotation
			coords = pd.DataFrame(turker['raw_data'][0])

			# Add the turker metadata to all entries in the data frame
			coords['annotation_type'] = turker['annotation_type']
			coords['height'] = turker['height']
			coords['width'] = turker['width']
			coords['image_filename'] = turker['image_filename']
			coords['time_when_completed'] = turker['time_when_completed']
			coords['worker_id'] = turker['worker_id']

			# Append to the total data frame
			self.annotations = self.annotations.append(coords)

	""" prints head of an annotation """
	def print_head(self):
		print(self.annotations.head(n=5))

	""" returns the entire pandas dataframe"""
	def df(self):
		return self.annotations

	""" returns the list of unique workers """
	def getWorkers(self):
		uid_series = self.annotations.worker_id
		return uid_series.unique()

	""" returns the list of unique image filenames """
	def getImages(self):
		img_series = self.annotations.image_filename
		return img_series.unique()

	""" 
	input: turker ID
	output: list with that turker's timestamps 
	"""
	def getTimestamps(self, uid):
		turker_df = self.getTurker(uid)
		turker_timestamps = turker_df.loc[:, ['timestamp']].as_matrix()
		return turker_timestamps

	""" 
	input: turker ID
	output: np array with coordinates of that turker's annotations 
	"""
	def getCoords(self, uid):
		turker_df = self.getTurker(uid)
		turker_coords = turker_df.loc[:, ['x', 'y']].as_matrix()
		return turker_coords

	""" 
	input: turker ID
	output: df with just that turker's annotations 
	"""
	def getTurker(self, uid):
		turker_df = self.annotations[self.annotations.worker_id == uid]
		return turker_df

	""" 
	inputs: 
		string name of clustering alg
		percent threshold of turkers required for valid clusters
	output:
		list of centroids
	"""
	def getClusters(self, clustering_alg, threshold):
		# TO DO
		pass

	""" 
	input: turker ID
	output: float avg time that turker spent per click 
	"""
	def get_avg_time_per_click(self, uid):
		turker_timestamps = self.getTimestamps(uid)
		time_spent = turker_timestamps[len(turker_timestamps)-1] - turker_timestamps[0]
		num_clicks = len(turker_timestamps)
		return time_spent[0]/num_clicks





















# ------------------------------ #




""" 
Input:
	string name of csv file containing reference points, aka "ground truth" values
Output:
	k-d tree containing the same reference points
"""
def csv_to_kdt(filename):

	ref_anno = pd.read_csv(filename)
	ref_points = ref_anno.loc[:, ['row', 'col']].as_matrix() - 1

	kdt = KDTree(ref_points, leaf_size=2, metric='euclidean')	# kdt is a kd tree with all the reference points

	return kdt

""" 
Inputs:
	k-d tree (reference points), aka "ground truth" values
	List of points (annotations) from many turkers
Output:
	List containing one list for each turker.
		Each list is comprised of, for each of the turker's
		points, the distance to the nearest neighbor (found in
		the k-d tree of references).
"""
def calc_distances(kdt, turker_anno, png_filename):

	to_return = []

	""" for each turker """
	for i in range(len(turker_anno)):

		turker = turker_anno[i]					# turker is a df

		if (turker.image_filename[0] == png_filename):

			turker_coords = turker.loc[:, ['x', 'y']].as_matrix()

			dist, ind = kdt.query(turker_coords, k=1)
			dist_list = dist.tolist()
			values = []

			for j in range(len(dist_list)):			# each element of dist_list is a list
				values.append(dist_list[j][0])

			to_return.append(values)

			""" plotting for sanity checking """

			# rand_point_ind = 4

			# fig = plt.figure()
			# axes = fig.add_subplot(111)
			# axes.scatter(turker_coords[:,0], turker_coords[:,1], s=10, c='b', label = 'turker_coords')
			# axes.scatter(ref[:,0], ref[:,1], s=10, c='r', label = 'ref')

			# axes.scatter(turker_coords[rand_point_ind,0], turker_coords[rand_point_ind,1], s=20, c='green')
			# axes.scatter(ref[ind[rand_point_ind],0], ref[ind[rand_point_ind],1], s=20, c='orange')

			# plt.legend(loc='upper left')
			# plt.show()
	return to_return



if __name__ == '__main__':

	csv_filename = 'bead_annotations_20180517.csv'
	json_filename = 'BeadAnnotation_20180413.json'
	png_filename = 'beads_300pxroi.png'

	kdt = csv_to_kdt(csv_filename)
	turker_anno = importAnnotations(json_filename)	# turker_anno is a list of DFs, one per turker

	fig = plt.figure()
	axes = fig.add_subplot(111)
	plt.ylabel('Avg dist betw click and nearest ref pt, time per click / 210')
	plt.xlabel('Turker')
	plt.title(png_filename)

	"""
	
	Average Distances

	"""

	list_of_distance_lists = calc_distances(kdt, turker_anno, png_filename)

	avg_list = []
	for i in range(len(list_of_distance_lists)):
		turker_list = list_of_distance_lists[i]
		avg = sum(turker_list)/float(len(turker_list))
		avg_list.append(avg)
	axes.scatter([range(0,len(avg_list))], avg_list)

	"""
	
	Time Spent

	"""

	times_spent_per_click_list = get_times_spent_per_click(turker_anno, png_filename)
	axes.scatter([range(0,len(times_spent_per_click_list))], times_spent_per_click_list)

	plt.show()

	

