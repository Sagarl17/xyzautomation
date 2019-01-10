import sys
import laspy
import pickle

def predict():
	'''
	Prediction from the pre trained and based on the features calculated
	  				for every point in point cloud
	'''
	filename = './models/model.sav'
	model = pickle.load(open(filename, 'rb'))
	test_data = np.load(('./data/interim/' + sys.argv[1])[:-4]+'_features.npy')	#training data
	logger.info("data loaded into memory")
	#test_data = np.delete(test_data, np.s_[15:21], axis=1)

	'''result = model.score(test_data[:,:-2], test_data[:,-1])
	logger.info(result)'''

	#processing data
	for i in range(len(test_data)):
		for j in range(test_data.shape[1]):
			if math.isnan(test_data[i][j]):
				test_data[i][j] = 0

	c = model.predict(test_data[:,:-1])
	logger.info("prediction done")
	logger.info("Saving file")
	infile = laspy.file.File('./data/raw/'+sys.argv[1], mode='rw')
	outfile = laspy.file.File('./data/interim/tobe_classified_'+sys.argv[1], mode="w", header=infile.header)
	outfile.points = infile.points
	outfile.classification = c.astype('uint32')
	infile.close()
	outfile.close()

if not(os.path.exists("./data/interim/tobe_classified_"+sys.argv[1])):
	predict()
