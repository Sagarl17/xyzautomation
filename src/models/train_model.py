import time
import os
import sys
import scipy
import math
import laspy
import psutil
import pickle
import logging
import numpy as np
import pandas as ps
import scipy.linalg
import datetime
import multiprocessing
import matplotlib as plt
from scipy import spatial
from sklearn import metrics
from numpy import linalg as LA
from xgboost import XGBClassifier
from sklearn.neighbors import KDTree
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def memcalc():
	mem='RAM: '+str(psutil.virtual_memory()[2])+'%'
	return mem

def cpucalc():
	cpu='CPU: '+str(psutil.cpu_percent(interval=None, percpu=False))+'%'
	return cpu

def setup_custom_logger(name):
	class ContextFilter(logging.Filter):
		def filter(self, record):
			record.memcalc = memcalc()
			record.cpucalc = cpucalc()
			return True



	formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(memcalc)s %(cpucalc)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )

	handler = logging.FileHandler('./logs/lOG_'+str(datetime.datetime.now()) +'.txt', mode='w')
	handler.setFormatter(formatter)
	screen_handler = logging.StreamHandler(stream=sys.stdout)
	screen_handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	logger.addHandler(handler)
	logger.addHandler(screen_handler)
	logger.addFilter(ContextFilter())
	return logger

logger = setup_custom_logger('myapp')


def neighbours(data, n=10):
	'''tree = KDTree(data[:,:])
	logger.info('KDTree built')
	_, idx = tree.query(data[:,:], k=n)
	return idx'''

	tree = spatial.KDTree(data)
	logger.info('KDTree built')
	_, idx = tree.query(data, k=10)
	return idx



class featurecalculation:
	def features(self,filename):
		"""
		INPUT :- LAS file name
		OUTPUT :- A numpy array of size (no. of points , 22) consisting predefined features
		"""
		pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)     # Create a multiprocessing Poolfor div in range(division):
		logger.info("calculating neighbours")
		result=pool.map(self.calc, range(division),chunksize=1)  # process data_inputs iterable with pool
		for divo in range(division):
			if divo == (division - 1):
				full_training_data[divo *maximum_points:] = result[divo][:][:]
			else :
				full_training_data[divo *maximum_points:(divo +1)*maximum_points] = result[divo][:][:]
			logger.info(divo)

		np.save('pointclouds/'+filename[:-4]+'_features' , full_training_data)

		return

	def calc(self,div):

		# Calculating Feature for small point cloud with (maximum_points) no. of points

		small_xyz = xyz[div*maximum_points:(div+1)*maximum_points]
		small_data = data[div*maximum_points:(div+1)*maximum_points]
		tree = spatial.KDTree(small_xyz)

		_, idx = tree.query(small_xyz[:,:], k=10)
		logger.info("Starting new Worker Process:%s",div)
		medoid = []
		for i in small_xyz[[idx]]:
			d = scipy.spatial.distance.pdist(i)
			d = scipy.spatial.distance.squareform(d)
			medoid.append(np.argmin(d.sum(axis=0)))

		covariance = []
		for i in small_xyz[[idx]]:
			covariance.append(np.cov(np.array(i).T))
		covariance = np.array(covariance)

		# Calculating Eigen Vectors and Eigen Values for each point
		# w: eigen values , v: eigen vectors
		w,v = LA.eigh(covariance)
		w = [i/np.sum(i) for i in w]
		w = np.array(w)

		training_data = np.zeros((len(small_xyz),22))

		# Calculating Geometric features for each point
		training_data[:,0] = np.power(np.multiply(np.multiply(w[:,0], w[:,1]), w[:,2]), 1/3)                                                    #omnivariance
		training_data[:,1] = -np.multiply(w[:,0], np.log(w[:,0]))-np.multiply(w[:,1], np.log(w[:,1]))-np.multiply(w[:,1], np.log(w[:,1]))       #eigenentropy
		training_data[:,2] = np.divide(w[:,2]-w[:,0], w[:,2])                                                                                   #anistropy
		training_data[:,3] =  np.divide(w[:,1]-w[:,0], w[:,2])                                                                                  #planarity
		training_data[:,4] =  np.divide(w[:,2]-w[:,1], w[:,2])                                                                                  #linearity
		training_data[:,5] = w[:,0]                                                                                                             #surface variation
		training_data[:,6] = np.divide(w[:,0], w[:,2])                                                                                          #scatter
		training_data[:,7] = 1-abs(v[:,0,2])                                                                                                    #verticality

		temp = []
		for i in range(len(small_xyz)):
			temp.append(np.subtract(small_xyz[idx[i]],small_xyz[idx[medoid[i]]]))

		# Calculating Central Moments and height feature for each point

		moment11 = []                   #moment 1st order 1st axis
		moment12 = []                   #moment 1st order 2nd axis
		moment21 = []                   #moment 2nd order 1st axis
		moment22 = []                   #moment 2nd order 2nd axis
		vertical_range = []             #vertical range
		height_below = []               #height below

		for i in range(len(small_xyz)):
			moment11.append(np.sum(np.dot(temp[i], v[i][2])))
			moment12.append(np.sum(np.dot(temp[i], v[i][1])))
			moment21.append((np.sum(np.dot(temp[i], v[i][2]))**2))
			moment22.append((np.sum(np.dot(temp[i], v[i][1]))**2))
			vertical_range.append((np.amax(small_xyz[idx[i]],axis=0))[2] - (np.amin(small_xyz[idx[i]],axis=0))[2])
			height_below.append(small_xyz[i][2] - (np.amin(small_xyz[idx[i]],axis=0))[2])

		training_data[:,8] = np.array(moment11)
		training_data[:,9] = np.array(moment12)
		training_data[:,10] = np.array(moment21)
		training_data[:,11] = np.array(moment22)
		training_data[:,12] = np.array(vertical_range)
		training_data[:,13] = np.array(height_below)
		moment11,moment12,moment21,moment22,temp = None,None,None,None,None

		#height above
		vertical_range = np.array(vertical_range)
		height_below = np.array(height_below)
		height_above = vertical_range - height_below
		training_data[:,14] = np.array(height_above)

		vertical_range,height_above,height_below = None,None,None

		# Calculating Color features for each points

		rgb2hsv = plt.colors.rgb_to_hsv((small_data[:,3:6]).astype('uint8'))
		training_data[:,15:18] = np.array(rgb2hsv)

		nbr_color = []
		for i in range(len(small_xyz)):
			nbr_color.append(np.sum(rgb2hsv[idx[i]], axis=0))
		nbr_color = np.array(nbr_color)
		nbr_color = nbr_color/10
		training_data[:,18:21] = np.array(nbr_color)

		nbr_color = None
		rgb2hsv = None

		y = small_data[:,6]
		training_data[:,21] = np.array(y)


		return training_data
for image_file_name in os.listdir('pointclouds'):
	if image_file_name.endswith(".las"):
		filename = image_file_name
		if not(os.path.exists("pointclouds/"+filename[:-4]+"_features.npy")):
			maximum_points = 50000
			infile = laspy.file.File("pointclouds/"+filename, mode='rw')
			print("Starting "+filename+" feature calculation")
			col = {'x':infile.x, 'y':infile.y, 'z':infile.z, 'r':infile.red/256, 'g':infile.green/256, 'b':infile.blue/256, 'c':infile.classification}
			data = ps.DataFrame(data=col)
			xyz = data.as_matrix(columns = ['x', 'y', 'z'])
			data = data.as_matrix(columns = ['x', 'y', 'z', 'r', 'g', 'b', 'c'])
			division = np.shape(xyz)[0]//maximum_points + 1
			full_training_data = np.zeros((np.shape(xyz)[0],22))

			fe=featurecalculation()
			fe.features(filename)
			infile.close()
			del col,data,xyz,full_training_data,division
			logger.info('features Calculation: Done')




for image_file_name in os.listdir('pointclouds'):
	if image_file_name.endswith(".npy"):
		filename = image_file_name
		#loading training data
		training_data=np.load("pointclouds/"+filename)
		logger.info("finished loading training data")
		#processing data
		for i in range(len(training_data)):
			for j in range(training_data.shape[1]):
				if math.isnan(training_data[i][j]):
					training_data[i][j] = 0
		logger.info("Finished Processing data")
		x_train, y_train = training_data[:,:-1],training_data[:,-1]

		class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
		if (os.path.exists("xgbmodel")):
			xgb = XGBClassifier(learning_rate =0.2,max_depth=16, n_estimators=600, subsample=0.5, colsample_bytree=0.5,objective= 'multi:softmax', nthread=3, scale_pos_weight=1, num_class=6,verbose_eval=True,xgb_model='xgbmodel.sav')
			xgb.fit(x_train,y_train)

			filename = './models/model.sav'
			pickle.dump(xgb, open(filename,'wb'))
			xgb.save_model("xgbmodel")
		else:
			xgb = XGBClassifier(learning_rate =0.2,max_depth=16, n_estimators=600, subsample=0.5, colsample_bytree=0.5,objective= 'multi:softmax', nthread=3, scale_pos_weight=1, num_class=6,verbose_eval=True)
			xgb.fit(x_train,y_train)

			filename = './models/model.sav'
			pickle.dump(xgb, open(filename,'wb'))
			xgb.save_model("xgbmodel")
