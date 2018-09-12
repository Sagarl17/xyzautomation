#############################################################################
#    				Importing Necessary Libraries                           #
## Usage python <full.py> <input LAS file>

import os
import sys
import time
import math
import json
import laspy
import scipy
import pickle
import logging
import datetime
import itertools
import shapefile
import numpy as np
import pandas as ps
import scipy.linalg
import random as rd
import matplotlib as plt
from sklearn import metrics
from numpy import linalg as LA
from pyproj import Proj,transform
from scipy import spatial,optimize
from sklearn.decomposition import PCA
from numpy.linalg import norm as LAnorm
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree,ConvexHull
from shapely.geometry import Polygon, mapping
from collections import defaultdict,OrderedDict
from shapely.ops import cascaded_union, polygonize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

rd.seed(236658149848658)
np.set_printoptions(threshold=np.nan)

# Cosine of Selected Angles
cos5 = 0.99619469809
cos10 = 0.98480775301
cos15 = 0.96592582628
cos20 = 0.93969262078
cos45 = 0.70710678118

def print_time():
	# For printing DATETIME
	return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

#####################################
# LAS Specifications
# Class 2 --> Ground
# Class 5 --> High Vegetation (aka Trees)
# Class 6 --> Buildings

#############################################################################
#     	Creating a Custom Logger for logging the process 					#

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger('myapp')

filename = str(sys.argv[1])

#############################################################################
#     	Calculating the features for each point in point cloud				#
#   For Fast Implementation point cloud is divided into set of points       #
#  And for set the features are then calculated and finally concatenated    #
#  		as a single numpy array with 22 features for each point 			#

maximum_points = 100000

def features(filename):	
	"""
	INPUT :- LAS file name
	OUTPUT :- A numpy array of size (no. of points , 22) consisting predefined features 
	"""
	
	infile = laspy.file.File(filename, mode='rw')
	col = {'x':infile.X, 'y':infile.Y, 'z':infile.Z, 'r':infile.red, 'g':infile.green, 'b':infile.blue, 'c':infile.classification}
	data = ps.DataFrame(data=col)

	logger.info("calculating neighbours")
	
	xyz = data.as_matrix(columns = ['x', 'y', 'z'])
	data = data.as_matrix(columns = ['x', 'y', 'z', 'r', 'g', 'b', 'c'])
	
	division = np.shape(xyz)[0]//maximum_points + 1
	
	full_training_data = np.zeros((np.shape(xyz)[0],22))
	
	logger.info(np.shape(xyz))
	
	for div in range(division):
		
		# Calculating Feature for small point cloud with (maximum_points) no. of points 
		
		small_xyz = xyz[div*maximum_points:(div+1)*maximum_points]
		small_data = data[div*maximum_points:(div+1)*maximum_points]
		
		tree = spatial.KDTree(small_xyz)
		
		_, idx = tree.query(small_xyz[:,:], k=10)

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
		training_data[:,0] = np.power(np.multiply(np.multiply(w[:,0], w[:,1]), w[:,2]), 1/3)													#omnivariance
		training_data[:,1] = -np.multiply(w[:,0], np.log(w[:,0]))-np.multiply(w[:,1], np.log(w[:,1]))-np.multiply(w[:,1], np.log(w[:,1]))		#eigenentropy
		training_data[:,2] = np.divide(w[:,2]-w[:,0], w[:,2])																					#anistropy
		training_data[:,3] =  np.divide(w[:,1]-w[:,0], w[:,2])																					#planarity
		training_data[:,4] =  np.divide(w[:,2]-w[:,1], w[:,2])																					#linearity
		training_data[:,5] = w[:,0]																												#surface variation
		training_data[:,6] = np.divide(w[:,0], w[:,2])																							#scatter
		training_data[:,7] = 1-abs(v[:,0,2])																									#verticality

		temp = []
		for i in range(len(small_xyz)):
			temp.append(np.subtract(small_xyz[idx[i]],small_xyz[idx[medoid[i]]]))
			
		# Calculating Central Moments and height feature for each point
		
		moment11 = []   				#moment 1st order 1st axis
		moment12 = []					#moment 1st order 2nd axis
		moment21 = []   				#moment 2nd order 1st axis
		moment22 = []   				#moment 2nd order 2nd axis
		vertical_range = []    			#vertical range				
		height_below = []  				#height below
				
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

		rgb2hsv = plt.colors.rgb_to_hsv((small_data[:,3:6]/256).astype('uint8'))
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
		
		if div == (division - 1):
			full_training_data[div*maximum_points:] = training_data
		else :
			full_training_data[div*maximum_points:(div+1)*maximum_points] = training_data
			
		logger.info(div)
		
	np.save(filename[:-4]+'_features' , full_training_data)
	
	logger.info("Feature Calculation Done")
	
	return



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#############################################################################
#  Adding a Specific Height Feature based on the planarity of the points    #
#  Repeatidely fitting a plane to the points in a particular plane and      #
#  giving height perspective                                                #


def add_height():
	infile = laspy.file.File(sys.argv[1],mode='rw')

	xyz = np.c_[infile.X, infile.Y, infile.Z]
	features = np.load((sys.argv[1])[:-4]+'_features.npy')

	k1 = len(xyz)/1000000.0
	k2 = float(np.amax(xyz[:,0])-np.amin(xyz[:,0]))/(np.amax(xyz[:,1])-np.amin(xyz[:,1]))
	scale_x = int(math.sqrt(k1*k2))
	scale_y = int(math.sqrt(k1/k2))
	c = np.zeros(len(xyz))
	
	print('scale:', scale_x, scale_y)

	A = np.c_[xyz[:,0], xyz[:,1], np.ones(xyz.shape[0])]
	P,_,_,_ = scipy.linalg.lstsq(A, xyz[:,2])

	data = []
	for i in range(len(xyz)):
		if xyz[i][2] - P[0]*xyz[i][0] - P[1]*xyz[i][1] - P[2] < 0:
			data.append(xyz[i])
	data = np.array(data)

	A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
	P,_,_,_ = scipy.linalg.lstsq(A, data[:,2])

	data1 = []
	for i in range(len(data)):
		if data[i][2] - P[0]*data[i][0] - P[1]*data[i][1] - P[2] < 0:
			data1.append(data[i])
	data1 = np.array(data1)

	A = np.c_[data1[:,0], data1[:,1], np.ones(data1.shape[0])]
	P,_,_,_ = scipy.linalg.lstsq(A, data1[:,2])
	new_plane = P

	distance = np.zeros(len(xyz))

	for x in range(scale_x):
		points = xyz[int(x*(len(xyz)/scale_x)):int((x+1)*(len(xyz)/scale_x)),:]
		sv = features[int(x*(len(xyz)/scale_x)):int((x+1)*(len(xyz)/scale_x)),5]
		idy = np.argsort(points[:,1], axis=0)
		for y in range(scale_y):
			grid_points = points[[idy[int(y*(len(points)/scale_y)):int((y+1)*(len(points)/scale_y))]]]
			grid_sv = sv[[idy[int(y*(len(points)/scale_y)):int((y+1)*(len(points)/scale_y))]]]
			temp = np.argsort(grid_points[:,2], axis=0)
			data = []
			count = 0
			mean = np.mean(grid_points[:int(len(grid_points)/2),2], axis=0)
			std = np.std(grid_points[:int(len(grid_points)/2),2], axis=0)
			for i in range(int(len(grid_points)/2)):
				p = grid_points[[temp[i]]]
				#if p[0][2]<mean+2*std and p[0][2]>mean-2*std and p[0][2] - P[0]*p[0][0] - P[1]*p[0][1] - P[2] < 0:
				if p[0][2] - P[0]*p[0][0] - P[1]*p[0][1] - P[2] < 0:
					p = grid_points[[temp[i]]]
					data.append(p[0])
					count += 1
			print(x,y)
			print(count)
			data = np.array(data)
			if count >=10 :
				A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
				new_plane,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
			else:
				print('not enough points to form a plane')


			if angle_between([P[0],P[1],1], [new_plane[0],new_plane[1],1]) < 0.174533:		#Angle < 10 degrees
				print(angle_between([P[0],P[1],1], [new_plane[0],new_plane[1],1]))
				C = new_plane
			else:
				print("false plane detected")
				if x!=0 and y==scale_y-1:
					C = back_plane
				elif x==0 and y==0:
					C = P

			if y == scale_y-1:
				back_plane = C

			for j in idy[int(y*(len(points)/scale_y)):int((y+1)*(len(points)/scale_y))]:
				k = math.sqrt(1 + C[0]**2 + C[1]**2)
				distance[int(x*(len(xyz)/scale_x))+j] = (xyz[int(x*(len(xyz)/scale_x))+j][2] - xyz[int(x*(len(xyz)/scale_x))+j][0]*C[0] - xyz[int(x*(len(xyz)/scale_x))+j][1]*C[1] - C[2])/k

	features = np.insert(features, -1, distance, axis=1)
	np.save('with_height_'+(sys.argv[1])[:-4]+'_features.npy', features)
	print("saved", features.shape)
	return

#############################################################################
#  Prediction from the pre trained and based on the features calculated     #
#    				for every point in point cloud							# 

def predict():
	filename = 'with_heights_eluru41_new_model_0.5_60.sav'
	model = pickle.load(open(filename, 'rb'))
	training_data = np.load('with_height_'+(sys.argv[1])[:-4]+'_features.npy')	#training data
	logger.info("data loaded into memory")
	#training_data = np.delete(training_data, np.s_[15:21], axis=1)

	'''result = model.score(training_data[:,:-2], training_data[:,-1])
	logger.info(result)'''

	#processing data
	for i in range(len(training_data)):
		for j in range(training_data.shape[1]):
			if math.isnan(training_data[i][j]):
				training_data[i][j] = 0

	c = model.predict(training_data[:,:-1])
	logger.info("prediction done")
	logger.info("Saving file")
	infile = laspy.file.File(sys.argv[1], mode='rw')
	outfile = laspy.file.File('classified_'+sys.argv[1], mode="w", header=infile.header)
	outfile.points = infile.points
	outfile.classification = c.astype('uint32')
	infile.close()
	outfile.close()

if not(os.path.exists("./"+filename[:-4]+"_features.npy")):
	features(filename)
	logger.info('features Calculation: Done')
if not(os.path.exists("./with_height_"+filename[:-4]+"_features.npy")):
	add_height()
	logger.info('Height feature added')
if not(os.path.exists("./classified_"+sys.argv[1])):
	predict()

#############################################################################
#		 for Training the Graadient Boosting model from scratch				#				

def training(filename):
	features(filename)
	print('learning_rate 0.5,  n_estimators 60')
	#loading training data
	training_data = np.load(filename[:-4]+'_features.npy')

	#processing data
	for i in range(len(training_data)):
		for j in range(training_data.shape[1]):
			if math.isnan(training_data[i][j]):
				training_data[i][j] = 0
	#x_train, x_test, y_train, y_test = train_test_split(training_data[:,:-1],training_data[:,-1], test_size=0.25, random_state=10)

	gbm = GradientBoostingClassifier(learning_rate=0.5, n_estimators=60, subsample=0.5, random_state=1, verbose=5, warm_start=True)
	gbm.fit(training_data[:,:-1],training_data[:,-1])

	filename = 'with_heights_eluru41_new_model_05_60.sav'
	pickle.dump(gbm, open(filename,'wb'))
	
	# Evaluating Model Performance
	'''
	result = gbm.score(training_data[:,:-1],training_data[:,-1])
	print(result)
	print('learning_rate 0.25,  n_estimators 120')
	'''
	return


logger.info('Trees Extraction: Started')

#############################################################################
#   	Converting the Z-coor to the actual height from the ground			#

input_file_name = 'classified_'+sys.argv[1]

clustering_threshold_radius = 0.6

def change_z_to_ht(input_file_name):
	infile=laspy.file.File('./'+input_file_name,mode='rw')
	point_3d=np.vstack([infile.X,infile.Y,infile.Z]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if classess[i]==2]
	main_header = infile.header
	outfile_name = "grounded_file.las"
	point_ground = point_3d[cand]
	#mean_height = np.mean(point_ground[:,2])
	pca = PCA(n_components=3)
	pca.fit(point_ground)
	normal = pca.components_[np.argmin(pca.explained_variance_)]
	if normal[2] < 0 :
		normal = -normal
	ht_ground  = []
	for i in point_ground:
		d = np.dot(i,normal)
		ht_ground.append(d)	
	ht_full = []
	for i in point_3d:
		d = np.dot(i,normal)
		ht_full.append(d)
	ht_full = np.array(ht_full)
	ht_full = ht_full-np.mean(ht_ground)
	outfile=laspy.file.File("./"+outfile_name,mode="w",header=main_header)
	outfile.points=infile.points
	outfile.Z = ht_full
	outfile.close()
	return outfile_name,main_header

if (os.path.exists("./grounded_file.las")):
	infile=laspy.file.File('./grounded_file.las',mode='rw')
	Header_for_all = infile.header
	ground_file_name = "grounded_file.las"
	logger.info('Found Grounded LAS File Already LoadingInfo...')
else :
	ground_file_name,Header_for_all = change_z_to_ht(input_file_name)
	logger.info('Grounded LAS File Created')


#############################################################################
# 			Converting the Full LAS file to trees only						#

def points_from_LAS(input_file_name,class_number_tree=5):
	infile=laspy.file.File('./'+input_file_name,mode='rw')
	point_3d=np.vstack([infile.X,infile.Y,infile.Z]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if classess[i]==class_number_tree]
	point_3d = np.take(infile.points,cand)
	
	outfile_name = "full_trees.las"
	outfile=laspy.file.File("./"+outfile_name,mode="w",header=Header_for_all)
	outfile.points=point_3d
	outfile.close()
	infile.close()
	return outfile_name

if (os.path.exists("./full_trees.las")):
	trees_file = "full_trees.las"
	logger.info('Found All Trees LAS File Already LoadingInfo...')
else :
	trees_file = points_from_LAS(ground_file_name)
	logger.info('All Trees LAS File Created')


#############################################################################
#      Removing the misclassified trees points with color constrained             #

def filter_green(trees_file):
	
	color_threshold = 25
	
	green_colors = [[124,252,0],[127,255,0],[50,205,50],[0,255,0],[34,139,34],[0,128,0],[0,100,0],[173,255,47],
					[154,205,50],[0,255,127],[0,250,154],[144,238,144],[152,251,152],[143,188,143],[60,179,113],
					[32,178,170],[46,139,87],[128,128,0],[85,107,47],[107,142,35]]
	
	infile=laspy.file.File('./'+trees_file,mode='rw')
	point_3d=np.vstack([infile.X,infile.Y,infile.Z]).T
	colors = np.vstack([infile.red,infile.green,infile.blue]).T
	colors = colors/256

	filtered_point = []
	for i in range(np.shape(colors)[0]):
		for j in green_colors:
			if abs(colors[i][0]-j[0])<color_threshold and abs(colors[i][1]-j[1])<color_threshold and abs(colors[i][2]-j[2])<color_threshold:
				filtered_point.append(i)
				break
	outfile_name = "trees_main_filtered.las"
	point_filt = np.take(infile.points,filtered_point)
	outfile=laspy.file.File("./"+outfile_name,mode="w",header=Header_for_all)
	outfile.points=point_filt
	outfile.close()
	infile.close()
	return outfile_name

if (os.path.exists("./"+"trees_main_filtered.las")):
	filtered_file = "trees_main_filtered.las"
	logger.info('Filtered Trees LAS File Already LoadingInfo...')
else :
	filtered_file = filter_green(trees_file)
	logger.info('Filtered Trees LAS File Created')

#############################################################################
#   Finding the Tree Tops as Seed for CLustering Based on Local Maximums    #


distance_b = 300

def tree_top_cand(filtered_file):
	infile = laspy.file.File('./'+filtered_file,mode='rw')
	point_3d = np.vstack([infile.X,infile.Y,infile.Z]).T
	scales = infile.header.scale
	offsets = infile.header.offset
	count_tree_tops = np.shape(point_3d)[0]//distance_b + 1
	int_tree_t = []
	for i in range(count_tree_tops):
		pts = point_3d[i*distance_b:(i+1)*distance_b]
		z_c = pts[:,2]
		local_max = np.argmax(z_c)
		int_tree_t.append(pts[local_max])
	infile.close()
	return int_tree_t,count_tree_tops,point_3d,scales,offsets

def dfs(adj_list, visited, vertex, result, key):
    visited.add(vertex)
    result[key].append(vertex)
    for neighbor in adj_list[vertex]:
        if neighbor not in visited:
            dfs(adj_list, visited, neighbor, result, key)

#############################################################################
#   	Merging the Tree Tops which are very close to each other	        #
  
def merging_adj_ttops(all_tree_top,no_intial_ttops):
	merging_ttop_dist = 1500
	tree_ttops = KDTree(np.asarray(all_tree_top)[:,0:2])
	b = tree_ttops.query_pairs(merging_ttop_dist)
	adj_list = defaultdict(list)
	for x, y in b:
		adj_list[x].append(y)
		adj_list[y].append(x)
		
	result = defaultdict(list)
	visited = set()
	for vertex in adj_list:
		if vertex not in visited:
			dfs(adj_list, visited, vertex, result, vertex)
	
	all_train = []
	for i in result.values():
		for j in i:
			all_train.append(j)
	a = np.array([x for x in range(no_intial_ttops)])
	remain = np.in1d(a,all_train,invert=True)
	remaining_clusters = a[remain]
	new_int_tree = []
	for j in remaining_clusters:
		new_int_tree.append(all_tree_top[j])
	for i in result.values():
		new_int_tree.append((np.asarray(all_tree_top)[i])[np.argmax((np.asarray(all_tree_top)[i])[:,2])])
	
	return new_int_tree

#############################################################################
#   	Clustering Based on the new tree tops and since there is a 	        #
#  correlation between height of the tree and radius of the tree using      #
#     		the clustering threshold in terms of height only                #

def getting_neighbour(new_ttops,point_3d):
	tree = KDTree(point_3d[:,0:2])
	neighbours = []
	point_completed = []
	for tree_top in new_ttops:
		pt_2d = tree_top[:2]
		height = tree_top[2]
		radius = height*clustering_threshold_radius
		neighbour = np.setdiff1d(tree.query_ball_point(pt_2d,radius),point_completed,assume_unique=True)
		neighbours.append(neighbour)
		point_completed.extend(neighbour)
	return neighbours

#############################################################################
# 		Saving the LAS file for better visulaization of different trees     #

def color_tree_las(neighbours,point_3d):
	color_dictionary = [[192,192,255],[192,192,128],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[192,192,192]
						,[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],[0,128,128],[0,0,128]]
	colors = np.shape(neighbours)[0]

	color_array = []
	for le in range(colors):
		color_array.append(color_dictionary[le%(len(color_dictionary))])

	X_coor,Y_coor,Z_coor = [],[],[]
	r,g,b = [],[],[]
	for i in range(len(neighbours)):
		if np.shape(point_3d[neighbours[i]])[0] > 100:
			array_data = np.asarray(point_3d[neighbours[i]],dtype="int")
			X_coor = np.concatenate((np.asarray(X_coor),np.asarray(array_data[:,0])),axis = 0)
			Y_coor = np.concatenate((np.asarray(Y_coor),np.asarray(array_data[:,1])),axis = 0)
			Z_coor = np.concatenate((np.asarray(Z_coor),np.asarray(array_data[:,2])),axis = 0)
			r = np.concatenate((np.asarray(r),np.full(np.shape(array_data)[0], color_array[i][0])),axis = 0)
			g = np.concatenate((np.asarray(g),np.full(np.shape(array_data)[0], color_array[i][1])),axis = 0)
			b = np.concatenate((np.asarray(b),np.full(np.shape(array_data)[0], color_array[i][2])),axis = 0)
	
	outfile=laspy.file.File("./trees_colored.las",mode="w", header=Header_for_all)
	outfile.X=X_coor
	outfile.Y=Y_coor
	outfile.Z=Z_coor
	outfile.red = np.array(r)
	outfile.green = np.array(g)
	outfile.blue = np.array(b)
	outfile.close()

#color_tree_las(neighbours,point_3d)
#logger.info('Individual Trees colored LAS file saved')

#############################################################################
# 		Saving the Trees Parameter in different Formats as Required         #

def tree_parameters_npy(neighbours,point_3d):
	polygons = []
	height_t = []
	for i in range(len(neighbours)):
		if np.shape(neighbours[i])[0] != 0 and np.shape(point_3d[neighbours[i]])[0]>10:
			array = point_3d[neighbours[i]]
			array = np.asarray(array)
			i_2d = array[:,0:2]
			ht_tree = np.max(array[:,2])
			hull = ConvexHull(i_2d)
			polygons.append(array[hull.vertices])
			height_t.append(ht_tree*scales[2]+offsets[2])
	all_data = np.array([polygons,height_t])
	all_data = np.asarray(all_data)
	#out_np_file = "tree_parameters"
	#np.save(out_np_file , all_data)
	return all_data

#############################################################################
#  Finding Radius from polygons and location as center for the trees        #

def calc_R(x,y,z, xc, yc, zc):
    return np.sqrt((x-xc)**2 + (y-yc)**2 + (z-zc)**2)

def f(c, x, y,z):
    Ri = calc_R(x, y,z, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y,z):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    z_m = np.mean(z)
    center_estimate = x_m, y_m, z_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y,z))
    xc, yc,zc = center
    Ri       = calc_R(x, y,z, *center)
    R        = Ri.mean()
    return R,center

#############################################################################
# 				Converting to LAT-LONG-Height coordinates                   #

def Proj_to_latlong(polygons,scales,offsets):
	
	inProj = Proj(init='epsg:32644')
	outProj = Proj(init='epsg:4326')
	location = []
	radius = []
	for i in polygons:
		r,c = leastsq_circle(i[:,0]*scales[0]+offsets[0],i[:,1]*scales[1]+offsets[1],i[:,2]*scales[2]+offsets[2])
		x,y = transform(inProj,outProj,c[0],c[1])
		cent = [x,y,c[2]]
		location.append(cent)
		radius.append(r)
		
	return radius,location

#############################################################################
# 				Shapefile for each tree					                    #

def shp_for_2D_polygon(polygons):
	multipoly_3D=[]
	for i in polygons:
		poly = Polygon(i)
		multipoly_3D.append(poly)

	w3D = shapefile.Writer(shapeType=shapefile.POLYGON)
	count_shp=0
	w3D.field("ID")
	w3D.autoBalance = 1
	for polygon in multipoly_3D:
		count_shp=count_shp+1
		polypar = []
		polygon = mapping(polygon)["coordinates"][0]
		for i in polygon:
			fro = []
			for dummy in i:
				fro.append(float(dummy))
			polypar.append(fro)
		w3D.poly(parts=[polypar])
		w3D.record(count_shp)
	w3D.save('polygons')

#shp_for_2D_polygon(polygons)
#logger.info('Trees 2D Polygon Shape file Saved')

#############################################################################
#   				Final Data in GeoJSON for MAPBOX API 					#

def trees_parameter_json(parameters_tree):
    features = []
    for i in range(np.shape(parameters_tree[1])[0]):
        data_dict = OrderedDict()
        data_dict["type"] = "Feature"
        new_dict = OrderedDict()
        new_dict["TreeHeight"] = float(parameters_tree[1][i])
        new_dict["TreeRadius"] = float(radius[i])
        data_dict["properties"]=  new_dict
        data_dict["geometry"] = OrderedDict(type = "Point",
                                    coordinates = location[i])
        features.append(data_dict)
                                               
    data = OrderedDict()
    data["type"] = "FeatureCollection"
    data["crs"] = OrderedDict(type = "name",properties = {"name":"urn:ogc:def:crs:OGC:1.3:CRS84"})
    data["features"] = features
    with open('tree_data.json', 'w') as f:
        json.dump(data,f)

if not(os.path.exists("./tree_data.json")):
	all_tree_top,no_intial_ttops,point_3d,scales,offsets = tree_top_cand(filtered_file)
	logger.info('Found All Trees top with local maximum')
	new_ttops = merging_adj_ttops(all_tree_top,no_intial_ttops)
	logger.info('Merged very close Tree Tops')
	neighbours = getting_neighbour(new_ttops,point_3d)
	logger.info('Found Neighbours of each Tree Top')
	parameters_tree = tree_parameters_npy(neighbours,point_3d)
	logger.info('Trees Parameter Calculated')
	polygons = parameters_tree[0]
	radius,location = Proj_to_latlong(polygons,scales,offsets)
	trees_parameter_json(parameters_tree)
	logger.info('JSON file with parameters Saved')

len_cube = 500

logger.info('RoofTop Extraction Started')

def building_LAS(ground_file_name):
	infile=laspy.file.File('./'+ground_file_name,mode='rw')
	point_3d=np.vstack([infile.X,infile.Y,infile.Z]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if classess[i]==6]
	point_to_store = np.take(infile.points,cand)
	point_to_return = point_3d[cand]
	outfile_name = "Buildings.las"
	outfile=laspy.file.File("./"+outfile_name,mode="w",header=infile.header)
	outfile.points=point_to_store
	outfile.close()
	
	return point_to_return,infile.header																			

if (os.path.exists("./Buildings.las")):
	infile=laspy.file.File('./Buildings.las',mode='rw')
	main_header = infile.header
	point_3d=np.vstack([infile.X,infile.Y,infile.Z]).T
	logger.info('Found Building LAS File Already LoadingInfo...')
else :
	point_3d,main_header = building_LAS(ground_file_name)
	logger.info('Building Points Extracted')

#######################################################################
#   Assigning a id to each point according to its respective cube     #

def normalize(point_3d,len_cube):
	min_arr = [min(point_3d[:,0]),min(point_3d[:,1]),min(point_3d[:,2])]
	max_arr = [max(point_3d[:,0]),max(point_3d[:,1]),max(point_3d[:,2])]
	shape_of_cubiod = np.subtract(max_arr,min_arr)
	range_of_cube = [shape_of_cubiod[0]//len_cube + 1,shape_of_cubiod[1]//len_cube + 1,shape_of_cubiod[2]//len_cube + 1]
	# Normalizing the points , shifting to origin
	point_3d[:,0] = point_3d[:,0] - (min_arr[0]-min_arr[0]%1000)
	point_3d[:,1] = point_3d[:,1] - (min_arr[1]-min_arr[1]%1000)
	point_3d[:,2] = point_3d[:,2] - (min_arr[2]-min_arr[2]%1000)
	
	normalization_array = [min_arr[0]-min_arr[0]%1000,min_arr[1]-min_arr[1]%1000,min_arr[2]-min_arr[2]%1000]
	
	return point_3d,range_of_cube,normalization_array
	
point_3d,range_of_cube,normalization_array = normalize(point_3d,len_cube)

def ID_points(point_3d,len_cube,range_of_cube):
		
	point_id = []
	dummy_point = point_3d[:][:]
	point_id = []
	for point in dummy_point:
		point_x = point[0]//len_cube
		point_y = point[1]//len_cube
		point_z = point[2]//len_cube
		point_id.append(point_x*range_of_cube[1]*range_of_cube[2]+point_y*range_of_cube[2]+point_z)
		

	unique_cube = sorted(list(set(point_id)))																

	#######################################################################
	#                Creating Dictionary for each unique ID               #

	diction = dict()
	count = 0 
	for i in point_id:
		if i in diction:
			diction[i].append(point_3d[count])
		else :
			diction[i]= [point_3d[count]]
		count += 1
	
	return unique_cube,diction
	
if not(os.path.exists("./all_"+str(len_cube)+".npy")):
	unique_cube,diction = ID_points(point_3d,len_cube,range_of_cube)
	logger.info('ID Assigned and Dictionary Created')

#######################################################################
# Doing the PCA and Saving Normals,Points Consisting and eigen vectors for plane  #

def PCA_cubes(unique_cube,range_of_cube,diction):
	step = 0
	normals = []
	indices_for_plane = []
	home_points_all = []

	for i in unique_cube:
		total_points = []
		index = [-1,0,1]
		permute = np.asarray([p for p in itertools.product(index, repeat=3)])
		surr_27_cubes = permute[:,0]*range_of_cube[1]*range_of_cube[2] + permute[:,1]*range_of_cube[2] + permute[:,2]
		surr_27_cubes += i
		points_in_cube = diction[i]
		home_point = np.squeeze(rd.sample(points_in_cube,1))
		for j in surr_27_cubes:
			if j in diction.keys():
				total_points += diction[j]
		total_points = np.asarray(total_points)
		distance_points = np.asarray(cdist(total_points,[home_point],'euclidean').flatten())
		indices_of_int = total_points[np.where(distance_points<len_cube)]
		num_points = np.shape(indices_of_int)[0]
		if num_points <= 10:
			continue
		else :
			pca = PCA(n_components=3)
			pca.fit(indices_of_int)
			
		normal = pca.components_[np.argmin(pca.explained_variance_)]
		step += 1
		
		normals.append(normal)
		indices_for_plane.append(indices_of_int)
		home_points_all.append(home_point)
	
	data = []
	data.append(normals)
	data.append(indices_for_plane)
	data.append(home_points_all)
	np.save("all_"+str(len_cube),data)
	
	return normals,indices_for_plane,home_points_all


if (os.path.exists("./all_"+str(len_cube)+".npy")):
	data = np.load("./all_"+str(len_cube)+".npy")
	normals = data[0]
	indices_for_plane = data[1]
	home_points_all = data[2]
	logger.info('Found PCA Parametrs Already LoadingInfo...')
else :
	normals,indices_for_plane,home_points_all = PCA_cubes(unique_cube,range_of_cube,diction)
	logger.info('PCA Done and parameters saved')

total_planes = np.shape(home_points_all)[0]

#######################################################################
#  Fitting the points to the plane and doing a Convex Hull for        #
#       getting exact shape of plane containing object                #
#       Saving the planes for 2D and 3D visualization                 #

def ConvexHull_planes(total_planes,normals,indices_for_plane,home_points_all):
	
	planes = []
	for i in range(total_planes):
		home_point = home_points_all[i]
		normal = normals[i]
		all_points = indices_for_plane[i]
		G_mat = all_points[:,0:2]
		G_mat.astype(float)
		mean = np.divide(np.sum(G_mat,axis = 0),np.shape(G_mat)[0])
		G_mat = G_mat - mean
		if np.shape(G_mat)[0] >2:
			hull = ConvexHull(G_mat)
		d = -home_point.dot(normal)
		G_mat = G_mat + mean
		z_coor = (-normal[0] * G_mat[:,0] - normal[1] * G_mat[:,1] - [d]*np.shape(G_mat)[0]) * 1. /normal[2]
		G_mat = np.hstack((G_mat,np.expand_dims(z_coor,axis=1)))
		planes.append(G_mat[hull.vertices])
	
	np.save("planes_"+str(len_cube),planes)
	
	return planes 

if (os.path.exists("./planes_"+str(len_cube)+".npy")):
	planes = np.load("./planes_"+str(len_cube)+".npy")
	logger.info('Found Planes Already LoadingInfo...')
else :
	planes = ConvexHull_planes(total_planes,normals,indices_for_plane,home_points_all)
	logger.info('Planes Calculated and Saved')

#######################################################################
#       Creating planes in 2D from 3D by removing Z-coordinate        #
#    Doing Cascade Union for intersecting plane for a consolidated    #
#    for Each Building and saving the shapefile for polygons created  #
#                       by above step                                 #

def shapefile_2D_roof(planes,main_header):
	two_d_planes = []
	scales = main_header.scale
	offsets = main_header.offset
	for i in planes:
		#new_array_0 = (i[:,0] + normalization_array[0])*scales[0]+offsets[0]
		#new_array_1 = (i[:,1] + normalization_array[1])*scales[1]+offsets[1]
		#two_d_planes.append(np.array([new_array_0,new_array_1]).tolist())
		two_d_planes.append(i[:,0:2])

	multipoly=[]
	for i in two_d_planes:
		poly = Polygon(i)
		multipoly.append(poly)
		
	unique_multipoly=cascaded_union(multipoly)

	w = shapefile.Writer(shapeType=shapefile.POLYGON)
	count=0
	w.field("ID")
	w.autoBalance = 1
	for polygon in unique_multipoly:
		count=count+1
		polypar = []
		polygon = mapping(polygon)["coordinates"][0]
		for i in polygon:
			fro = []
			for dummy in i:
				fro.append(float(dummy))
			polypar.append(fro)
		w.poly(parts=[polypar])
		w.record(count)
	w.save('planes_2D_'+str(len_cube))

if (os.path.exists('./planes_2D_'+str(len_cube)+'.shp')):
	logger.info('Saved the Shape File Already')
else :
	shapefile_2D_roof(planes,main_header)
	logger.info('Saved 2D shapefiles for Rooftops')

#######################################################################
#       Loading 3D planes and doing Cascading Union for joining       #
#       intersecting planes and saving it for future                  #
"""
def polygon_3D(planes,main_header):
	
	scales = main_header.scale
	offsets = main_header.offset
	
	multipoly3D=[]
	for i in planes:
		new_array_0 = (i[:,0] + normalization_array[0])*scales[0]+offsets[0]
		new_array_1 = (i[:,1] + normalization_array[1])*scales[1]+offsets[1]
		new_array_2 = (i[:,2] + normalization_array[2])*scales[2]+offsets[2]
		i = np.array([new_array_0,new_array_1,new_array_2]).tolist()
		poly = Polygon(i)
		multipoly3D.append(poly)
		
	unique_polys3D=cascaded_union(multipoly3D)

	#######################################################################
	#        Converting the polygons to the vertices of a plane           # 
	
	final_planes_3D = []
	for poly in unique_polys3D:
		polypar = []
		polygon = mapping(poly)["coordinates"][0]
		for i in polygon:
			fro = []
			for dummy in i:
				fro.append(float(dummy))
			polypar.append(fro)
		final_planes_3D.append(polypar)

	#######################################################################
	#   Calculating the min height and the max height for each plane      #
	heights = np.array([])
	for i in final_planes_3D:
		i = np.asarray(i)
		max_x,max_y,max_z = i.max(axis=0)
		min_x,min_y,min_z = i.min(axis=0)
		heights = np.append(heights,np.array([min_z,max_z]))
	
	heights = np.reshape(heights,(-1,2))
	
	np.save("final_planes_3D_"+str(len_cube),final_planes_3D)
	np.save("heights_"+str(len_cube),heights)
	
	return final_planes_3D,heights
	

if (os.path.exists("./final_planes_3D_"+str(len_cube)+".npy")):
	final_planes_3D = np.load("./final_planes_3D_"+str(len_cube)+".npy")
	heights = np.load("./heights_"+str(len_cube)+".npy")
	print ( print_time() , " 3D Planes Parameters Found Already LoadingInfo...")
else :
	final_planes_3D,heights = polygon_3D(planes)
	print ( print_time() , " 3D Planes Parameter found and Saved")

#######################################################################
#        Saving the 3D planes coordinates and heights(min,max)        #
 
def parameter_json(final_planes_3D,heights):
	final_planes_3D = np.asarray(final_planes_3D,)
	final_dict = {}
	for i in range(np.shape(final_planes_3D)[0]):
		dummy_a = np.asarray(final_planes_3D[i],dtype=int)
		dummy_b = np.asarray(heights[i],dtype=int)
		data_dict = dict(planes_coor_3D = np.expand_dims(dummy_a,axis=0).tolist(),
						 heights = np.expand_dims(dummy_b,axis=0).tolist())
		final_dict["Plane_No-"+str(i)] = data_dict
	with open('polygon_data.json', 'w') as f:
		json.dump(final_dict,f,indent=2,sort_keys=True)

parameter_json(final_planes_3D,heights)
"""
 


logger.info('Building Plane Extraction Started')

#############################################################################
# 			Converting the Full LAS file to building only					#

if (os.path.exists("./Buildings.las")):
	infile = laspy.file.File('./Buildings.las',mode='rw')
	point_3d = np.vstack([infile.X,infile.Y,infile.Z]).T
	main_header = infile.header
	logger.info('Found Building LAS File Already LoadingInfo...')
else :
	point_3d,main_header = building_LAS(input_file_name)
	logger.info('Building Points Extracted')

#############################################################################
# 		Finding Normal and Curvature for each point of building				#

def normals_curvature(point_3d):
	
	maximum_points = 10000
	division = np.shape(point_3d)[0]//maximum_points + 1
	normals,curvatures = np.zeros((np.shape(point_3d)[0],3)),np.zeros((np.shape(point_3d)[0],))
	
	for div in range(division):
		small_cluster = point_3d[div*maximum_points:(div+1)*maximum_points]
		tree_small = KDTree(small_cluster)
		_, idx = tree_small.query(small_cluster[:,:], k=8)
		covariance = []
		for i in small_cluster[[idx]]:
			covariance.append(np.cov(np.array(i).T))
		covariance = np.array(covariance)
		w,v = LA.eigh(covariance)
		w = [i[0]/np.sum(i) for i in w]
		w = np.array(w)
		v = np.array(v)
		v = v[:,:,0]
		normals[div*maximum_points:(div+1)*maximum_points] = v
		curvatures[div*maximum_points:(div+1)*maximum_points] = w
		logger.info(div)

	data_normal = []
	data_normal.append(normals)
	data_normal.append(np.expand_dims(curvatures,axis=0))
	
	np.save("normal_cur_main",data_normal)
	
	return normals,curvatures

if (os.path.exists("./normal_cur_main.npy")):
	logger.info(" Normal_Curvature data , Loading....")
	data_normals = np.load("normal_cur_main.npy")
	normals = np.asarray(data_normals[0])
	curvatures = np.asarray(data_normals[1][0],dtype="float64")
else :
	normals,curvatures = normals_curvature(point_3d)
	logger.info("Normals and Curvature Found and Saved")

total_points = np.shape(normals)[0]

#############################################################################
# 	Clustering based on the minimum curvature point (Recursive Seeding)		#

def region_clustering(point_3d,normals,curvatures):
	
	regions = []
	regions_not_considered = []
	indices_for_regions = []
	indices_for_regions_n = []

	count_region = 0
	major_region = 0
	# Maximum Points to be Clustered at once 
	maximum_points = 10000
	cur_threshold = 0.15
	
	# Dividing the Clustering into small patches of points so that agglomerative clustering can be made faster
	
	division = np.shape(point_3d)[0]//maximum_points
	logger.info("Clustering Started")
	
	for div in range(division):
		removed_points = []
		dummy_curvatures = curvatures[div*maximum_points:(div+1)*maximum_points]
		new_curvatures = curvatures[div*maximum_points:(div+1)*maximum_points]
		new_normals = normals[div*maximum_points:(div+1)*maximum_points]
		new_points = point_3d[div*maximum_points:(div+1)*maximum_points]
		tree = KDTree(new_points)
		while(len(removed_points) < maximum_points):
			count_region += 1
			curvature_update = np.delete(dummy_curvatures,removed_points)
			lowest_value = np.min(curvature_update)
			lowest_index = np.where(new_curvatures==lowest_value)[0][0]
			# If Normal is not reliable , i.e. cur>cur_threshold then don't do clustering for that point
			if new_curvatures[lowest_index] > cur_threshold:
				removed_points.append(lowest_index)
				continue
			
			indices_for_regions_i = []
			seeded_region_i = []
			regions_i = []
			seeded_region_i.append(lowest_index)
			regions_i.append(tree.data[lowest_index])
			indices_for_regions_i.append(lowest_index)
			removed_points.append(lowest_index)
			
			for k in seeded_region_i:
				previous = 30
				ind_30 = tree.query(tree.data[k],previous)
				indexes = ind_30[1][:-1]
				common_points = np.intersect1d(indexes,removed_points,assume_unique=True)
				indexes = np.setdiff1d(indexes,common_points)
				
			# For Each Neighbour Check the Criteria of Merging and Assign a new SEED if Curvature is small enough 	 
				for j in indexes:
					cos_dis = (np.dot(new_normals[j],new_normals[k])/(LAnorm(new_normals[j])*LAnorm(new_normals[k])))
					if abs(cos_dis) > cos10:
						regions_i.append(tree.data[j])
						indices_for_regions_i.append(j)
						removed_points.append(j)
						if (new_curvatures[j] < cur_threshold):
							seeded_region_i.append(j)
			# Major Regions are considered as planes having atleast 100 points
			if (len(regions_i) > 100):
				regions.append(regions_i)
				indices_for_regions.append(indices_for_regions_i)
				major_region += 1 
			# Other Regions are saved as Minor Regions
			else:
				regions_not_considered.append(regions_i)
				indices_for_regions_n.append(indices_for_regions_i)
			
		print (print_time() , " Division #:- %d , #Major Regions = %d,#All Regions = %d" % (div,major_region,count_region))
	logger.info("Clustering Done")
	logger.info("Major Regions")
	logger.info(np.shape(regions)[0])
	logger.info("Minor Regions")
	logger.info(np.shape(regions_not_considered)[0])
	
	# Saving Data for Use in Future
	data_region = []
	data_region.append(regions)
	data_region.append(indices_for_regions)
	data_region.append(regions_not_considered)
	data_region.append(indices_for_regions_n)
	
	logger.info("Regions File Saved")
	np.save("regions_main",data_region)

	return regions,indices_for_regions,regions_not_considered,indices_for_regions_n
	
	
if (os.path.exists("./regions_main.npy")):
	logger.info("Found Regions! Loading.....")
	data_regions = np.load("regions_main.npy")
	regions = data_regions[0]
	indices_for_regions = data_regions[1]
	regions_not_considered = data_regions[2]
	indices_for_regions_n = data_regions[3]
else :
	regions,indices_for_regions,regions_not_considered,indices_for_regions_n = region_clustering(point_3d,normals,curvatures)

def combining_planes(old_normals_major,full_regions):
	joining_thre = 15000
	point_thre_combine = 200
	normals_major = []
	regions = []
	for i in range(np.shape(full_regions)[0]):
		if (np.shape(full_regions[i])[0]) > point_thre_combine:
			regions.append(full_regions[i])
			normals_major.append(old_normals_major[i])
	
	hull_vertices = []
	
	for j in regions:
		j = np.asarray(j)
		dummy_j = j[:,0:2]
		hull_j = ConvexHull(dummy_j)
		hull_vertices.append(j[hull_j.vertices])
		
	angle_matrix = -(cdist(normals_major,normals_major,metric='cosine')-1)
	joined = []
	new_regions = []
	all_indices = [i for i in range(np.shape(angle_matrix)[0])]
	#angle_matrix = np.absolute(angle_matrix)
	for i in all_indices:
		if i in joined:
			continue
		else:
			joined.append(i)
			indexes = np.where(abs(angle_matrix[i])>cos15)[0]
			mask = np.in1d(indexes, joined, invert=True , assume_unique=True)
			regions_remaining = indexes[mask]
			#regions_remaining = np.intersect1d(a,joined,assume_unique=True)
			start = regions[i]
			for j in regions_remaining:
				b = np.min(cdist(hull_vertices[i],hull_vertices[j]))
				if b < joining_thre :
					start = np.concatenate((np.asarray(start),np.asarray(regions[j])),axis=0)
					joined.append(j)
			new_regions.append(start)
	logger.info("Combined very close Planes")

	return new_regions


def adding_minor_regions(regions_not_considered,regions,normals_major):
	
	total_points_not = 0
	
	for j in regions_not_considered:
		j = np.asarray(j,dtype="float64")
		total_points_not += np.shape(j[:,0])[0]
	
	print ("# of Pts in Minor Regions = %d" % total_points_not)

	count_minor_regions = 0
	planes_added = 0

	joining_thre_min = 500
	# Joining the regions based on the minimum distance and angle between the region and regions normal respectively
	
	for j in regions_not_considered:
		count_minor_regions += 1
		j = np.asarray(j,dtype="float64")
		if (np.shape(j)[0]==1):
			for i in range(np.shape(normals_major)[0]):
				a = np.asarray(regions[i])
				if np.min(cdist(a,j)) < joining_thre:
					regions[i] = np.concatenate((np.asarray(regions[i]),j))
		elif (np.shape(j)[0]==2):
			a = np.asarray(regions[i])
			for i in range(np.shape(normals_major)[0]):
				if np.min(cdist(a,j)) < joining_thre:
					regions[i] = np.concatenate((np.asarray(regions[i]),j))
		else:
			pca_not = PCA(n_components=3)
			pca_not.fit(j)
			normal_not = pca_not.components_[np.argmin(pca_not.explained_variance_)]
			for i in range(np.shape(normals_major)[0]):
				cos_dis = (np.dot(normal_not,normals_major[i])/(LAnorm(normal_not)*LAnorm(normals_major[i])))
				if abs(cos_dis) > cos5:
					a = np.asarray(regions[i])
					if np.min(cdist(a,j)) < joining_thre:
						regions[i] = np.concatenate((np.asarray(regions[i]),j))
						planes_added += 1 
						break
		if ((count_minor_regions % 500)==0):
			print (print_time() , " # Minor Planes Covered :- ",count_minor_regions,"# of Minor Planes added :- ",planes_added, "Time Taken = " ,time.time()-time5)

	total_points_join = 0

	for i in regions:
		i = np.asarray(i,dtype="float64")
		total_points_join += np.shape(i[:,0])[0]
	print ("# of Pts Covered after Joining = %d" % total_points_join)
	
	return regions

angle_thre = cos45
point_thre = 400 

def fitting_plane(regions,regions_not_considered):
	
	total_points = 0
	planes_final = []
	normals_major = []

	# Fitting a plane to the point in One Region
	for i in regions:
		i = np.asarray(i,dtype="float64")
		total_points += np.shape(i[:,0])[0]
		pca2 = PCA(n_components=3)
		pca2.fit(i)
		planes_normal = pca2.components_[np.argmin(pca2.explained_variance_)]
		normals_major.append(planes_normal)
	
	logger.info("# of Pts in Major Regions")
	logger.info(total_points)
	
	# Adding minor regions to the major regions with threshold (experimentally chosen)
	# regions_minandmajor = adding_minor_regions(regions_not_considered,regions,normals_major)
	
	# Combining regions which are close enough and have very less difference in there normals
	regions = combining_planes(normals_major,regions)
	
	new_regions = []
	
	# Removing the planes which have points less than a point_threshold
	# Removing the planes which are along the wall (As Need to detect Rooftop features only) by angle_threshold
	# Convex Hull Method from Scipy is Used for getting minimum area Boundaries of a region
	
	for j in range(len(regions)):
		k = regions[j]
		k = np.asarray(k,dtype="float64")
		pca3 = PCA(n_components=3)
		pca3.fit(k)
		planes_n2 = pca3.components_[np.argmin(pca3.explained_variance_)]
		midpoint_k = np.divide(np.sum(k,axis=0),np.shape(k)[0])
		d = -midpoint_k.dot(planes_n2)
		#z_coor = i[:,2]
		k = k[:,0:2]
		hull = ConvexHull(k)
		z_coor = (-planes_n2[0] * k[:,0] - planes_n2[1] * k[:,1] - [d]*np.shape(k)[0]) * 1. /planes_n2[2]
		k = np.hstack((k,np.expand_dims(z_coor,axis=1)))
		angle_y = (np.dot(planes_n2,[0,1,0])/LAnorm(planes_n2))
		angle_x = (np.dot(planes_n2,[1,0,0])/LAnorm(planes_n2))
		if abs(angle_y) < angle_thre and abs(angle_x) < angle_thre and np.shape(k)[0]>point_thre:
			new_regions.append(k)
			planes_final.append(k[hull.vertices])

	data_final = []
	data_final.append(new_regions)
	data_final.append(planes_final)

	np.save("final_regions_planes_pca",data_final)
	logger.info("Final Planes Saved")
	
	return new_regions,planes_final


if (os.path.exists("./final_regions_planes_pca.npy")):
	logger.info("Found Final Regions! Loading.....")
	data_regions = np.load("final_regions_planes_pca.npy")
	new_regions = data_regions[0]
	planes_final = data_regions[1]
else :
	new_regions,planes_final = fitting_plane(regions,regions_not_considered)
	logger.info("Final Regions and 2D planes File Saved")

def shapefile_2D_planes(planes_final,main_header):
	
	two_d_planes = []
	scales = main_header.scale
	offsets = main_header.offset
	
	for i in planes_final:
		#new_array_0 = (i[:,0] + normalization_array[0])*scales[0]+offsets[0]
		#new_array_1 = (i[:,1] + normalization_array[1])*scales[1]+offsets[1]
		#two_d_planes.append(np.array([new_array_0,new_array_1]).tolist())
		two_d_planes.append(i[:,0:2])
		
	multipoly=[]
	for i in two_d_planes:
		poly = Polygon(i)
		multipoly.append(poly)
		

	w = shapefile.Writer(shapeType=shapefile.POLYGON)
	count_shp=0
	w.field("ID")
	w.autoBalance = 1
	for polygon in multipoly:
		count_shp=count_shp+1
		polypar = []
		polygon = mapping(polygon)["coordinates"][0]
		for i in polygon:
			fro = []
			for dummy in i:
				fro.append(float(dummy))
			polypar.append(fro)
		w.poly(parts=[polypar])
		w.record(count_shp)
	w.save('plane_shapes')
	logger.info("2D shape file saved")

if not(os.path.exists('plane_shapes.shp')):
	shapefile_2D_planes(planes_final,main_header)


def LAS_file_planes(new_regions):
	color_dictionary = [[192,192,255],[192,192,128],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],
						[192,192,192],[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],[0,128,128],[0,0,128]]
	colors = np.shape(new_regions)[0]

	color_array = []
	for le in range(colors):
		color_array.append(color_dictionary[le%(len(color_dictionary))])

	X_coor,Y_coor,Z_coor = [],[],[]
	r,g,b = [],[],[]
	for i in range(len(new_regions)):
		array_data = np.asarray(new_regions[i],dtype="int")
		X_coor = np.concatenate((np.asarray(X_coor),np.asarray(array_data[:,0])),axis = 0)
		Y_coor = np.concatenate((np.asarray(Y_coor),np.asarray(array_data[:,1])),axis = 0)
		Z_coor = np.concatenate((np.asarray(Z_coor),np.asarray(array_data[:,2])),axis = 0)
		r = np.concatenate((np.asarray(r),np.full(np.shape(array_data)[0], color_array[i][0])),axis = 0)
		g = np.concatenate((np.asarray(g),np.full(np.shape(array_data)[0], color_array[i][1])),axis = 0)
		b = np.concatenate((np.asarray(b),np.full(np.shape(array_data)[0], color_array[i][2])),axis = 0)
	  
	outfile=laspy.file.File("./plane_"+str(angle_thre)+"_"+str(point_thre)+"_pca.las",mode="w", header=main_header)
	outfile.X=X_coor
	outfile.Y=Y_coor
	outfile.Z=Z_coor
	outfile.red = np.array(r)
	outfile.green = np.array(g)
	outfile.blue = np.array(b)
	outfile.close()
	logger.info("planes LAS file saved")
	
#if not(os.path.exists("plane_"+str(angle_thre)+"_"+str(point_thre)+"_pca.las")):
#	LAS_file_planes(new_regions)

def plane_parameter_json(new_regions,planes_final):
	
	final_dict = {}
	for i in range(np.shape(new_regions)[0]):
		dummy_a = np.array(new_regions[i],dtype=int)
		dummy_b = np.asarray(planes_final[i],dtype=int)
		data_dict = dict(lower_height = str(np.min(dummy_a[:,2])),
						 upper_height = str(np.max(dummy_a[:,2])),
						 polygon_vertices = np.expand_dims(dummy_b,axis=0).tolist())
		final_dict["Plane_ID"+str(i)] = data_dict
	with open('planes_data.json', 'w') as f:
		json.dump(final_dict,f,indent=2,sort_keys=True)
		
		
if not(os.path.exists('planes_data.json')):
	plane_parameter_json(new_regions,planes_final)
	logger.info("JSON file with parameters Saved")



logger.info('Building CAD file creation started')


def poly_to_CAD(planes):
	
	
	f = open("planes_CAD.off","a")
	f.write("OFF\n")

	total_points = 0
	all_points = []
	for i in planes:
		total_points += np.shape(i)[0]
		for j in i:
			all_points.append(np.array(j,dtype=int))

	no_polygons = (np.shape(planes)[0])*2 + total_points

	f.write(str(2*total_points)+" "+str(no_polygons)+" 0\n")

	for i in all_points:
		f.write(str(i[0])+" "+str(i[1])+" "+str(i[2])+"\n")
	for i in all_points:
		f.write(str(i[0])+" "+str(i[1])+" "+str(0)+"\n")

	number = 0
	for i in planes:
		array = str(np.shape(i)[0])+" "
		for j in range(len(i)):
			array += str(j+number)+" "
		array = array[:-1]+"\n"
		f.write(array)
		number += np.shape(i)[0]
		
	number = 0
	for i in planes:
		array = str(np.shape(i)[0])+" "
		for j in range(len(i)):
			array += str(j+number+total_points)+" "
		array = array[:-1]+"\n"
		f.write(array)
		number += np.shape(i)[0]

	number = 0
	for i in range(len(planes)):
		no_point_in_plane = np.shape(planes[i])[0]
		for j in range(no_point_in_plane):
			if j==no_point_in_plane-1:
				to_write = "4 "+str(j+number)+" "+str(number)+" "+str(total_points+number)+" "+str(j+total_points+number)+"\n"
			else :
				to_write = "4 "+str(j+number)+" "+str(j+1+number)+" "+str(j+1+total_points+number)+" "+str(j+total_points+number)+"\n"
			f.write(to_write)
		number += np.shape(planes[i])[0]

	f.close()

if not(os.path.exists('planes_CAD.off')):
	poly_to_CAD(planes_final)
	logger.info('CAD file saved')
