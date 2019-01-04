def training(filename):
	''' training the gradient boosted tree '''
	f=featurecalculation()
	f.features(filename)
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
