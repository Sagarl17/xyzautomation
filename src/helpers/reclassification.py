import laspy
import sys
import os
import numpy as np


def rc():
	#building section
	infile=laspy.file.File('./data/interim/tobe_classified_'+sys.argv[1],mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z,infile.red,infile.green,infile.blue]).T
	classess=infile.classification
	cin = [i for i in range(len(point_3d)) if classess[i]==6]
	cand = [i for i in range(len(point_3d)) if ((classess[i]==6) and (point_3d[i,4]<0.45*point_3d[i,3]+0.65*point_3d[i,5]))]
	building_points = np.take(infile.points,cand)
	cpn=list(set(cin)-set(cand))
	point_to_classify1 =np.take(infile.points,cpn)
	#trees section
	cin = [i for i in range(len(point_3d)) if classess[i]==5]
	cand= [i for i in range(len(point_3d)) if classess[i]==5 and (point_3d[i,4]>=0.45*point_3d[i,3]+0.65*point_3d[i,5])]

	trees_points = np.take(infile.points,cand)
	cin=list(set(cin)-set(cand))
	point_to_classify2 =np.take(infile.points,cin)

	#roads section
	cin = [i for i in range(len(point_3d)) if classess[i]==11]
	cand = [i for i in range(len(point_3d)) if ((classess[i]==11) and (point_3d[i,4]<0.45*point_3d[i,3]+0.65*point_3d[i,5]))]

	road_points=np.take(infile.points,cand)
	cin=list(set(cin)-set(cand))
	point_to_classify3 =np.take(infile.points,cin)

    #ground section

	cin = [i for i in range(len(point_3d)) if classess[i]==2]
	cand = [i for i in range(len(point_3d)) if ((classess[i]==2) and (point_3d[i,4]<0.45*point_3d[i,3]+0.65*point_3d[i,5]))]

	ground_points=np.take(infile.points,cand)
	cin=list(set(cin)-set(cand))
	point_to_classify4 =np.take(infile.points,cin)

	point_to_classify=np.append(point_to_classify1,point_to_classify2)
	point_to_classify5=np.append(point_to_classify4,point_to_classify3)
	point_to_classify7=np.append(point_to_classify,point_to_classify5)
	
	pts=np.append(building_points,trees_points)
	pts2=np.append(road_points,ground_points)
	classified_points=np.append(pts,pts2)


	outfile=laspy.file.File('./data/interim/classified_'+sys.argv[1],mode="w",header=infile.header)
	outfile.points=point_to_classify7
	point_3d=np.vstack([outfile.x,outfile.y,outfile.z,outfile.red,outfile.green,outfile.blue,outfile.classification]).T
	classess=outfile.classification
	for i in range(len(point_3d)):
		if point_3d[i,4]>=0.39*point_3d[i,3]+0.61*point_3d[i,5]:
			point_3d[i,6]=5
		elif (40<point_3d[i,3]/256<125 and 30<point_3d[i,4]/256<93 and 21<point_3d[i,5]/256<65 ):
			point_3d[i,6]=2
		elif (point_3d[i,3]/256<30 and point_3d[i,4]/256<30 and point_3d[i,5]/256<30):
			point_3d[i,6]=11
		else:
			point_3d[i,6]=6
	outfile.x=point_3d[:,0]
	outfile.y=point_3d[:,1]
	outfile.z=point_3d[:,2]
	outfile.Red=point_3d[:,3]
	outfile.Green=point_3d[:,4]
	outfile.Blue=point_3d[:,5]
	cl=[]
	for i in point_3d[:,6]:
		cl.append(int(i))
	outfile.classification=cl
	point_3d=np.vstack([outfile.x,outfile.y,outfile.z,outfile.red,outfile.green,outfile.blue,outfile.classification]).T
	outfile.close()


	outfile=laspy.file.File('./data/interim/classified_'+sys.argv[1],mode="w",header=infile.header)
	outfile.points=classified_points
	point_3d1=np.vstack([outfile.x,outfile.y,outfile.z,outfile.red,outfile.green,outfile.blue,outfile.classification]).T
	outfile.close()
	point_3d=np.concatenate((point_3d,point_3d1),axis=0)
	outfile=laspy.file.File('./data/interim/classified_'+sys.argv[1],mode="w",header=infile.header)
	outfile.x=point_3d[:,0]
	outfile.y=point_3d[:,1]
	outfile.z=point_3d[:,2]
	outfile.Red=point_3d[:,3]
	outfile.Green=point_3d[:,4]
	outfile.Blue=point_3d[:,5]
	cl=[]
	for i in point_3d[:,6]:
		cl.append(int(i))
	outfile.classification=cl
	outfile.close()
	return

