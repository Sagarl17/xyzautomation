import os
import json
import math
import laspy
import geojson
import shapely
import progressbar
import numpy as np
from scipy.spatial import KDTree,ConvexHull
from shapely.geometry import Polygon,mapping,Point

def road_LAS(ground_file_name):
	infile=laspy.file.File('./'+ground_file_name,mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z,infile.red,infile.green,infile.blue]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if classess[i]==11]
	point_to_store = np.take(infile.points,cand)
	point_to_return = point_3d[cand]
	outfile_name = "roads.las"
	outfile=laspy.file.File("./data/processed/"+outfile_name,mode="w",header=infile.header)
	outfile.points=point_to_store
	outfile.close()

	return point_to_return,infile.header

def Clustering():
	infile=laspy.file.File('./data/processed/roads.las',mode='rw')
	main_header = infile.header
	point_3d=np.vstack([infile.x,infile.y,infile.z]).T
	d=len(point_3d)

	tree = KDTree(point_3d)                                       # Create KDtree representation of pointcloud                                          
	Clusters=[] 
	points=[]
	intensity=[]
	FC=set()                                               # Intialize empty cluster list
	Q=[]
	i=1
	bar = progressbar.ProgressBar(maxval=len(point_3d), widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	for point in range(len(point_3d)):
		l1=len(FC)
		FC.add(point)
		l2=len(FC)
		if l2>l1:
			Q.append(point)
			p1=point_3d[point]
			for point in Q:
				NP=tree.query_ball_point(point_3d[point],r = 0.175)
				for point1 in NP:
					p2=point_3d[point1]
					if math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))<=5:
						l1=len(FC)
						FC.add(point1)
						l2=len(FC)
						if l2>l1:
							Q.append(point1)
							bar.update(len(FC))
						
			F=[]					
			if len(Q)>500:
				point_to_store = np.take(infile.points,Q)
				x=np.ones(len(point_to_store))*i
				if len(points)==0:
					
					points=point_to_store
					intensity=x
				else:
					points=np.append(points,point_to_store)
					intensity=np.append(intensity,x)
				i=i+1
				F=F+Q
			Q=[]
	outfile=laspy.file.File("Rest.las",mode="w",header=infile.header)
	outfile.points=points
	outfile.close()
	infile=laspy.file.File("Rest.las",mode="r")

	header = infile.header
	outfile=laspy.file.File("./data/processed/Roads_Clustered.las",mode="w",header=header)
	outfile.define_new_dimension(name = "cluster_id",data_type = 5, description = "Cluster_id")
	outfile.cluster_id=intensity
	outfile.x = infile.x
	outfile.y = infile.y
	outfile.z = infile.z
	outfile.red = infile.red
	outfile.green = infile.green
	outfile.blue = infile.blue
	bar.finish()
	outfile.close()
	os.remove("Rest.las")

def Polygonextraction():
	infile = laspy.file.File('./data/processed/Roads_Clustered.las',mode ='rw')
	arr =[]
	inten = max(infile.cluster_id)
	print(inten)

	final = {"type":"FeatureCollection", "features": []}

	for i in range(1,inten+1):
		arr =[]
		feature ={"type":"Feature","geometry":{"type":"Polygon","coordinates":[]},"properties":{"id":i}}
		print(i)
		clusterx = infile.x[infile.cluster_id ==i]
		clustery = infile.y[infile.cluster_id ==i]

		points =np.vstack([clusterx,clustery]).T
	
		hull = ConvexHull(points)
		for i in hull.vertices:
			arr.append([points[i,0],points[i,1]])

		feature['geometry']['coordinates'].append(arr)
		final['features'].append(feature)


	with open('./data/processed/Roads_data.json', 'w') as outfile:
		json.dump(final, outfile)

def Mergingpolygons():
	with open('./data/processed/Roads_data.json') as geojson1:
		poly1_geojson = json.load(geojson1)
	poly=[]


	for i in range(len(poly1_geojson['features'])):
		poly.append(shapely.geometry.asShape(poly1_geojson['features'][i]['geometry']))
	index=list(range(0,len(poly)))
	# checking to make sure they registered as polygons
	count=1
	while(count<11):
		print(count*10)
		index=list(range(0,len(poly)))
		for i in index:
			for j in index:
				if ((poly[i].intersects(poly[j])==True or poly[i].within(poly[j]) or poly[j].within(poly[i])or poly[i].crosses(poly[j])==True or poly[i].distance(poly[j])<1.5 or poly[i].touches(poly[j])==True ) and i!=j ):
					poly[i]=poly[i].union(poly[j])
					del poly[j]
					index.remove(j)
					for n, k in enumerate(index):
						if k>j:
							index[n] = index[n]-1
		count=count+1



	final = {"type":"FeatureCollection", "features": []}
	for i in range(len(poly)):
			geojson_out=geojson.Feature(geometry=poly[i])
			feature ={"type":"Feature","geometry":{"type":"Polygon","coordinates":[]},"properties":{"id":i}}
			feature['geometry']=geojson_out.geometry
			final['features'].append(feature)

	with open('./data/external/Roads.json', 'w') as outfile:
		json.dump(final, outfile)