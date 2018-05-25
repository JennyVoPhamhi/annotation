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

	

