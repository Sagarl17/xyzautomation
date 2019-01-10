import laspy
import numpy as np

def ground_LAS(ground_file_name):
	infile=laspy.file.File('./'+ground_file_name,mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z,infile.red,infile.green,infile.blue]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if classess[i]==2]

	point_to_store = np.take(infile.points,cand)
	point_to_return = point_3d[cand]
	outfile_name = "ground.las"
	outfile=laspy.file.File("./data/processed/"+outfile_name,mode="w",header=infile.header)
	outfile.points=point_to_store
	outfile.close()

	return point_to_return,infile.header
