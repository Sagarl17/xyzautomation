
import random as rd
import numpy as np
from src.helpers import logger,z2ht,vegetation,buildings,roads,ground,reclassification

import laspy
import pandas as ps
import sys
import os

rd.seed(236658149848658)
np.set_printoptions(threshold=np.nan)
sys.setrecursionlimit(100000)

logger = logger.setup_custom_logger('myapp')

filename = str(sys.argv[1])

###########################################################################
# feature calculation

logger.info("feature Calculation: Started")

exec(open('./src/features/build_features.py').read())

logger.info('features Calculation: Done')

############################################################################
# prediction

logger.info("prediction : Started")

exec(open('./src/models/predict_model.py').read())

logger.info('prediction : Done')

############################################################################
#reclassifying misclassified points based on characterstics of points

logger.info("Reclassifying : Started")
if not(os.path.exists("./data/interim/classified_"+sys.argv[1])):
	reclassification.rc()

logger.info('Reclassifying : Done')

############################################################################
# converting the z-coord to actual height above the ground

input_file_name = './data/interim/classified_'+sys.argv[1]



if (os.path.exists("./data/interim/grounded_file.las")):
	infile=laspy.file.File('./data/interim/grounded_file.las',mode='rw')
	Header_for_all = infile.header
	ground_file_name = "./data/interim/grounded_file.las"
	logger.info('Found Grounded LAS File Already LoadingInfo...')
else :
	ground_file_name,Header_for_all = z2ht.change_z_to_ht(input_file_name)
	logger.info('Grounded LAS File Created')

if ("trees" in sys.argv):
	############################################################################
	# trees extraction

	logger.info("Trees Extraction : Started")

	if (os.path.exists("./data/processed/full_trees.las")):
		trees_file = "full_trees.las"
		logger.info('Found All Trees LAS File Already LoadingInfo...')
	else :
		trees_file = vegetation.points_from_LAS(ground_file_name)
		logger.info('All Trees LAS File Created')


	logger.info("Trees Extraction : Done")

	############################################################################
	#Getting Json files of trees with trees as points

	distance_b = 300

	if not(os.path.exists("./data/external/tree_data.json")):
		all_tree_top,no_intial_ttops,point_3d,scales,offsets = vegetation.tree_top_cand(trees_file)
		logger.info('Found All Trees top with local maximum')
		new_ttops = vegetation.merging_adj_ttops(all_tree_top,no_intial_ttops)
		logger.info('Merged very close Tree Tops')
		neighbours = vegetation.getting_neighbour(new_ttops,point_3d)
		logger.info('Found Neighbours of each Tree Top')
		parameters_tree = vegetation.tree_parameters_npy(neighbours,point_3d,scales,offsets)
		logger.info('Trees Parameter Calculated')
		polygons = parameters_tree[0]
		radius,location = vegetation.Proj_to_latlong(polygons,scales,offsets)
		vegetation.trees_parameter_json(parameters_tree,radius,location)
		logger.info('JSON file with parameters Saved')

if ("buildings" in sys.argv):
	############################################################################
	#Extracting Building Points

	logger.info("Building Extraction : Started")

	if (os.path.exists("./data/processed/buildings.las")):
		infile=laspy.file.File('./data/processed/buildings.las',mode='rw')
		main_header = infile.header
		point_3d=np.vstack([infile.x,infile.y,infile.z]).T
		logger.info('Found Building LAS File Already LoadingInfo...')
	else :
		point_3d,main_header = buildings.building_LAS(ground_file_name)
		logger.info('Building Points Extracted')

	logger.info("Building Extraction : Finished")

	###########################################################################
	#Clustering process for Buildings

	logger.info("Building Clustering : Started")

	if (os.path.exists("./data/interim/Clustered.las")):
		logger.info('Clustering Already Done')
	else :
		buildings.Clustering()
		logger.info('Clustering Extracted')

	logger.info("Building Clustering : Finished")

	###########################################################################
	#Creating Polygons and getting json file

	logger.info("Creating Polygons : Started")

	if (os.path.exists("./data/interim/Buildings_data.json")):
		logger.info('Building polygons already extracted')
	else :
		buildings.Polygonextraction()
		logger.info('Polygon Extraction Completed')

	logger.info("Creating Polygons : Finished")


	###########################################################################
	#Merging Polygons and getting json file

	logger.info("Merging Polygons : Started")

	if (os.path.exists("./data/external/Merged_Buildings_data.json")):
		logger.info('Building polygons already merged')
	else :
		buildings.Mergingpolygons()
		logger.info('Polygons merged')

	logger.info("Merging Polygons : Finished")

#################################################################################

if ("roads" in sys.argv):
	############################################################################
	#Extracting Road Points

	logger.info("Road Extraction : Started")

	if (os.path.exists("./data/processed/roads.las")):
		infile=laspy.file.File('./data/processed/roads.las',mode='rw')
		main_header = infile.header
		point_3d=np.vstack([infile.x,infile.y,infile.z]).T
		logger.info('Found Road LAS File Already LoadingInfo...')
	else :
		point_3d,main_header = roads.road_LAS(ground_file_name)
		logger.info('Road Points Extracted')

	logger.info("Road Extraction : Finished")



if ("ground" in sys.argv):
	############################################################################
	#Extracting Ground Points

	logger.info("Ground Extraction : Started")

	if (os.path.exists("./data/processed/ground.las")):
		infile=laspy.file.File('./data/processed/ground.las',mode='rw')
		main_header = infile.header
		point_3d=np.vstack([infile.x,infile.y,infile.z]).T
		logger.info('Found Ground LAS File Already LoadingInfo...')
	else :
		point_3d,main_header = ground.ground_LAS(ground_file_name)
		logger.info('Ground Points Extracted')

	logger.info("Ground Extraction : Finished")


