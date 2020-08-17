import os 
import sys
import math
import laspy
import numpy as np
import progressbar
import multiprocessing
from scipy.spatial import KDTree


sys.setrecursionlimit(10000)
filename=sys.argv[1]


class featurecalculation:
	def features(self,infilepoints,header):
		"""
		INPUT :- LAS file name
		OUTPUT :- A numpy array of size (no. of points , 22) consisting predefined features
		"""
		size=[]
		heights=[]
		F=[]
		points=[]
		intensity=[]
		Height=[]
		pool = multiprocessing.Pool(processes=division)     # Create a multiprocessing Poolfor div in range(division):
		result=pool.map(self.calc, range(division),chunksize=1)  # process data_inputs iterable with pool
		for divo in range(division):
			F=F+result[divo][0]
			size=size+result[divo][1]
			heights=heights+result[divo][2]



		for i in range(len(size)):
			x=F[:size[i]]
			del F[:size[i]]
			point_to_store = np.take(infilepoints,x)
			f=np.ones(len(point_to_store))*i
			h=np.ones(len(point_to_store))*heights[i]
			if len(points)==0:
				
				points=point_to_store
				intensity=f
				Height=h
			else:
				points=np.append(points,point_to_store)
				intensity=np.append(intensity,f)
				Height=np.append(Height,h)
		outfile=laspy.file.File("Test.las",mode="w",header=header)
		outfile.points=points
		outfile.close()
		infile=laspy.file.File("Test.las",mode="r")

		header = infile.header
		outfile=laspy.file.File('./data/processed/Trees_Clustered_'+filename,mode='w',header=header)
		outfile.define_new_dimension(name = "cluster_id",data_type = 5, description = "Cluster_id")
		outfile.define_new_dimension(name = "height",data_type = 10, description = "Height")
		outfile.cluster_id=intensity
		outfile.height=Height
		outfile.x = infile.x
		outfile.y = infile.y
		outfile.z = infile.z
		outfile.red = infile.red
		outfile.green = infile.green
		outfile.blue = infile.blue



		outfile.close()
		os.remove("Test.las")

		return

	def calc(self,div):
		Clusters=[] 
		points=[]
		intensity=[]
		i=1
		FC=set()                                               # Intialize empty cluster list
		Q=[]
		F=[]
		size=[]
		heights=[]
		index=list(range(div*maximum_points,min((div+1)*maximum_points,len(point_3d))))
		for point in index:
			l1=len(FC)
			FC.add(point)
			l2=len(FC)
			if l2>l1:
				Q.append(point)
				p1=point_3d[point]
				for point in Q:
					NP=tree.query_ball_point(point_3d[point],r = .15)
					for point1 in NP:
						if point1 in index:
							p2=point_3d[point1]
							if math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))<=7.5:
								l1=len(FC)
								FC.add(point1)
								l2=len(FC)
								if l2>l1:
									Q.append(point1)
								
									
				if len(Q)>50:
					size.append(len(Q))
					F=F+Q
					point_to_store = np.take(point_3d,Q,axis=0)
					minx =np.amin(point_to_store[:,2])
					maxx =np.amax(point_to_store[:,2])
					point=point_to_store[np.where(point_to_store[:,2] == minx)]
					n=tree2.query_ball_point(point,r=10)
					n=list(n[0])
					if len(n)==0:
						h=maxx-minx
					else:
						cp=np.take(point_3d2,n,axis=0)
						mint =np.amin(cp[:,2])
						h=maxx-mint
					heights.append(h)
				Q=[]

			
		return F,size,heights








infile=laspy.file.File('./data/processed/full_trees_'+filename,mode='rw')
infile2=laspy.file.File('./data/raw/'+filename,mode='rw')
header = infile.header
point_3d=np.vstack([infile.x,infile.y,infile.z]).T
point_3d2=np.vstack([infile2.x,infile2.y,infile2.z]).T
th=np.vstack([infile.z]).T
ph=np.vstack([infile2.z]).T
ifp=infile.points
division=multiprocessing.cpu_count()
maximum_points=(len(point_3d)//division)+1

tree = KDTree(point_3d)                                # Create KDtree representation of pointcloud
tree2= KDTree(point_3d2)  
fe=featurecalculation()
fe.features(ifp,header)
