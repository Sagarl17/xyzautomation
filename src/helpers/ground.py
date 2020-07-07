import laspy
import numpy as np

def ground_LAS(filename):
	infile=laspy.file.File('./data/interim/grounded_file_'+filename,mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z,infile.red,infile.green,infile.blue]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if classess[i]==2]

	point_to_store = np.take(infile.points,cand)
	point_to_return = point_3d[cand]
	outfile=laspy.file.File("./data/processed/grounded_"+outfile_name,mode="w",header=infile.header)
	outfile.points=point_to_store
	outfile.close()

	return point_to_return,infile.header
