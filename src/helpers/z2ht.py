import laspy
import numpy as np
from sklearn.decomposition import PCA

def change_z_to_ht(input_file_name):
	infile=laspy.file.File(input_file_name,mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z]).T
	classess=infile.Classification
	cand = [i for i in range(len(point_3d)) if classess[i]==2]
	main_header = infile.header
	outfile_name = "data/interim/grounded_file.las"
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
	outfile=laspy.file.File(outfile_name,mode="w",header=main_header)
	outfile.points=infile.points
	outfile.z = ht_full
	outfile.close()
	return outfile_name,main_header
