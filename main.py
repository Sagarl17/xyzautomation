
import random as rd
import numpy as np
from src.helpers import logger,z2ht,vegetation,buildings,roads,ground,reclassification
import progressbar
import laspy
import sys
import os

rd.seed(236658149848658)
np.set_printoptions(threshold=sys.maxsize)
sys.setrecursionlimit(100000)

logger = logger.setup_custom_logger('myapp')

filename = str(sys.argv[1])

###########################################################################
# feature calculation

logger.info('feature Calculation: Started')

exec(open('./src/features/build_features.py').read())

logger.info('features Calculation: Done')

############################################################################
# prediction

logger.info('prediction : Started')

exec(open('./src/models/predict_model.py').read())

logger.info('prediction : Done')

############################################################################
#reclassifying misclassified points based on characterstics of points

logger.info('Reclassifying : Started')
if not(os.path.exists('./data/interim/classified_'+filename)):
	reclassification.rc()

logger.info('Reclassifying : Done')

############################################################################
# converting the z-coord to actual height above the ground




if (os.path.exists('./data/interim/grounded_file_'+filename)):
	infile=laspy.file.File('./data/interim/grounded_file_'+filename,mode='rw')
	Header_for_all = infile.header
	ground_file_name = './data/interim/grounded_file_'+filename
	logger.info('Found Grounded LAS File Already LoadingInfo...')
else :
	ground_file_name,Header_for_all = z2ht.change_z_to_ht(filename)
	logger.info('Grounded LAS File Created')

if ('trees' in sys.argv):
	############################################################################
	# trees extraction

	logger.info('Trees Extraction : Started')

	if (os.path.exists('./data/processed/full_trees_'+filename)):
		logger.info('Found All Trees LAS File Already LoadingInfo...')
	else :
		vegetation.vegetation_LAS(filename)
		logger.info('All Trees LAS File Created')


	logger.info('Trees Extraction : Done')
	###########################################################################
	#Clustering process for Buildings

	logger.info('Trees Clustering : Started')

	if (os.path.exists('./data/processed/Trees_Clustered_'+filename)):
		logger.info('Clustering Already Done')
	else :
		exec(open('./src/helpers/veg_cluster.py').read())
		logger.info('Clustering Extracted')

	logger.info('Trees Clustering : Finished')

	###########################################################################
	#Creating Polygons and getting json file

	logger.info('Creating Points : Started')

	if (os.path.exists('./data/interim/Trees_data_'+filename[:-4]+'.json')):
		logger.info('Tree points already extracted')
	else :
		exec(open('./src/helpers/veg_poly.py').read())
		logger.info('Point Extraction Completed')

	logger.info('Creating Points : Finished')


	###########################################################################
	#Merging Polygons and getting json file

	logger.info('Merging Points : Started')

	if (os.path.exists('./data/external/Trees_'+filename[:-4]+'.json')):
		logger.info('Tree points already merged')
	else :
		vegetation.Mergingpolygons(filename)
		logger.info('Tree Points merged')

	logger.info('Merging Points : Finished')

#################################################################################

#################################################################################

	

if ('buildings' in sys.argv):
	############################################################################
	#Extracting Building Points

	logger.info('Building Extraction : Started')

	if (os.path.exists('./data/processed/buildings_'+filename)):
		infile=laspy.file.File('./data/processed/buildings.las',mode='rw')
		logger.info('Found Building LAS File Already LoadingInfo...')
	else :
		buildings.building_LAS(filename)
		logger.info('Building Points Extracted')

	logger.info('Building Extraction : Finished')

	###########################################################################
	#Clustering process for Buildings

	logger.info('Building Clustering : Started')

	if (os.path.exists('./data/processed/Buildings_Clustered_'+filename)):
		logger.info('Clustering Already Done')
	else :
		buildings.Clustering(filename)
		logger.info('Clustering Extracted')

	logger.info('Building Clustering : Finished')

	###########################################################################
	#Creating Polygons and getting json file

	logger.info('Creating Polygons : Started')

	if (os.path.exists('./data/interim/Buildings_data_'+filename[:-4]+'.json')):
		logger.info('Building polygons already extracted')
	else :
		buildings.Polygonextraction(filename)
		logger.info('Polygon Extraction Completed')

	logger.info('Creating Polygons : Finished')


	###########################################################################
	#Merging Polygons and getting json file

	logger.info('Merging Polygons : Started')

	if (os.path.exists('./data/external/Buildings_'+filename[:-4]+'.json')):
		logger.info('Building polygons already merged')
	else :
		buildings.Mergingpolygons(filename)
		logger.info('Polygons merged')

	logger.info('Merging Polygons : Finished')

#################################################################################

if ('roads' in sys.argv):
	############################################################################
	#Extracting Road Points

	logger.info('Road Extraction : Started')

	if (os.path.exists("./data/processed/roads_"+filename)):
		logger.info('Found Road LAS File Already LoadingInfo...')
	else :
		roads.road_LAS(filename)
		logger.info('Road Points Extracted')

	logger.info('Road Extraction : Finished')

		###########################################################################
	#Clustering process for Roads

	logger.info('Road Clustering : Started')

	if (os.path.exists("./data/processed/Roads_Clustered_"+filename)):
		logger.info('Clustering Already Done')
	else :
		roads.Clustering(filename)
		logger.info('Clustering Extracted')

	logger.info('Road Clustering : Finished')

	###########################################################################
	#Creating Polygons and getting json file

	logger.info('Creating Polygons : Started')

	if (os.path.exists('./data/processed/Roads_data_'+filename[:-4]+'.json')):
		logger.info('Road polygons already extracted')
	else :
		roads.Polygonextraction(filename)
		logger.info('Polygon Extraction Completed')

	logger.info('Creating Polygons : Finished')


	###########################################################################
	#Merging Polygons and getting json file

	logger.info('Merging Polygons : Started')

	if (os.path.exists('./data/external/Roads_'+filename[:-4]+'.json')):
		logger.info('Road polygons already merged')
	else :
		roads.Mergingpolygons(filename)
		logger.info('Polygons merged')

	logger.info('Merging Polygons : Finished')

#################################################################################



if ('ground' in sys.argv):
	############################################################################
	#Extracting Ground Points

	logger.info('Ground Extraction : Started')

	if (os.path.exists('./data/processed/ground.las')):
		infile=laspy.file.File('./data/processed/ground.las',mode='rw')
		logger.info('Found Ground LAS File Already LoadingInfo...')
	else :
		point_3d,main_header = ground.ground_LAS(filename)
		logger.info('Ground Points Extracted')

	logger.info('Ground Extraction : Finished')


