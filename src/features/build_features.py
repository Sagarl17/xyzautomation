import os
import sys
import math
import laspy
import scipy
import numpy as np
import pandas as ps
import scipy.linalg
import multiprocessing
import matplotlib as plt
from numpy import linalg as LA
from scipy import spatial,optimize
from sklearn.decomposition import PCA




filename = str(sys.argv[1])

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
		
		np.save('./data/interim/'+filename[:-4]+'_features' , full_training_data)

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

		training_data = np.zeros((len(small_xyz),21))

		# Calculating Geometric features for each point
		training_data[:,0] = np.power(np.multiply(np.multiply(w[:,0], w[:,1]), w[:,2]), 1/3)                                                    #omnivariance
		training_data[:,1] = -np.multiply(w[:,0], np.log(w[:,0]))-np.multiply(w[:,1], np.log(w[:,1]))-np.multiply(w[:,2], np.log(w[:,2]))       #eigenentropy
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
		return training_data




if not(os.path.exists("./data/interim/"+filename[:-4]+"_features.npy")):

	infile = laspy.file.File("./data/raw/"+filename, mode='rw')
	col = {'x':infile.x, 'y':infile.y, 'z':infile.z, 'r':infile.red/256, 'g':infile.green/256, 'b':infile.blue/256, 'c':infile.classification}
	data = ps.DataFrame(data=col)
	xyz=data[['x', 'y', 'z']].to_numpy()
	data=data[['x', 'y', 'z', 'r', 'g', 'b', 'c']].to_numpy()
	maximum_points=np.shape(xyz)[0]//(multiprocessing.cpu_count()-1)+1
	division = np.shape(xyz)[0]//maximum_points + 1
	full_training_data = np.zeros((np.shape(xyz)[0],21))
	fe=featurecalculation()
	fe.features(filename)
	infile.close()
	del col,data,xyz,full_training_data,division
	logger.info('features Calculation: Done')
