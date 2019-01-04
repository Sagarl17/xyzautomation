import numpy as np
import laspy

def road_LAS(ground_file_name):
	infile=laspy.file.File('./'+ground_file_name,mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z,infile.red,infile.green,infile.blue]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if ((classess[i]==11 and ((point_3d[i,4]<0.39*point_3d[i,3]+0.61*point_3d[i,5]) and ((.85*point_3d[i,4]<point_3d[i,3])and(.7*point_3d[i,4]<point_3d[i,5])))))]
	point_to_store = np.take(infile.points,cand)
	point_to_return = point_3d[cand]
	outfile_name = "Roads.las"
	outfile=laspy.file.File("./"+outfile_name,mode="w",header=infile.header)
	outfile.points=point_to_store
	outfile.close()

	return point_to_return,infile.header
