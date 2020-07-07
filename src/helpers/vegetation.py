import os
import json
import math
import laspy
import geojson
import shapely
import progressbar
import numpy as np
from scipy.spatial import KDTree,ConvexHull
from shapely.geometry import Polygon,mapping,Point,LineString


def vegetation_LAS(filename):
	'''
	extracting the tree points from the classified point cloud
	'''
	infile=laspy.file.File('./data/interim/grounded_file_'+filename,mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if classess[i]==5]
	point_3d = np.take(infile.points,cand)
	outfile=laspy.file.File('./data/processed/full_trees_'+filename,mode='w',header=infile.header)
	outfile.points=point_3d
	outfile.close()
	infile.close()

def Clustering(filename):
	infile=laspy.file.File('./data/processed/full_trees_'+filename,mode='rw')
	infile2=laspy.file.File('./data/raw/'+filename,mode='rw')
	main_header = infile.header
	point_3d=np.vstack([infile.x,infile.y,infile.z]).T
	point_3d2=np.vstack([infile2.x,infile2.y,infile2.z]).T
	th=np.vstack([infile.z]).T
	ph=np.vstack([infile2.z]).T
	d=len(point_3d)

	tree = KDTree(point_3d)                                # Create KDtree representation of pointcloud
	tree2= KDTree(point_3d2)                                       
	Clusters=[] 
	points=[]
	intensity=[]
	FC=set()                                               # Intialize empty cluster list
	Q=[]
	i=1
	c=0
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
				NP=tree.query_ball_point(point_3d[point],r = .175)
				for point1 in NP:
					p2=point_3d[point1]
					if (abs(p1[0]-p2[0])<=10 or abs(p2[1]-p1[1])<=10 ):
						l1=len(FC)
						FC.add(point1)
						l2=len(FC)
						if l2>l1:
							Q.append(point1)
							bar.update(len(FC))
							
						
			F=[]					
			if len(Q)>100:	
				point_to_store = np.take(point_3d,Q,axis=0)
				minx =np.amin(point_to_store[:,2])
				maxx =np.amax(point_to_store[:,2])
				point=point_to_store[np.where(point_to_store[:,2] == maxx)]
				n=tree2.query_ball_point(point,r=10)
				n=list(n[0])
				if len(n)==0:
					h=maxx-minx
					point_to_store = np.take(infile.points,Q)
					s=np.ones(len(point_to_store))*h
					x=np.ones(len(point_to_store))*i
				else:
					cp=np.take(point_3d2,n,axis=0)
					mint =np.amin(cp[:,2])
					h=maxx-mint
					point_to_store = np.take(infile.points,Q)
					s=np.ones(len(point_to_store))*h
					x=np.ones(len(point_to_store))*i
				if len(points)==0:
					
					points=point_to_store
					intensity=x
					synthetic=s
				else:
					points=np.append(points,point_to_store)
					intensity=np.append(intensity,x)
					synthetic=np.append(synthetic,s)
				i=i+1
				F=F+Q
			Q=[]

	FC=list(FC)
	outfile=laspy.file.File('Test.las',mode='w',header=infile.header)
	outfile.points=points
	outfile.close()
	infile=laspy.file.File('Test.las',mode='r')

	header = infile.header
	outfile=laspy.file.File('./data/processed/Trees_Clustered_'+filename,mode='w',header=header)
	outfile.define_new_dimension(name = 'height',data_type = 10, description = 'Height')
	outfile.define_new_dimension(name = 'cluster_id',data_type = 5, description = 'Cluster_id')
	outfile.cluster_id=intensity
	outfile.height = synthetic
	outfile.x = infile.x
	outfile.y = infile.y
	outfile.z = infile.z
	outfile.red = infile.red
	outfile.green = infile.green
	outfile.blue = infile.blue
	bar.finish()
	outfile.close()
	os.remove('Test.las')


def Polygonextraction(filename):
	infile = laspy.file.File('./data/processed/Trees_Clustered_'+filename,mode ='rw')
	arr=[]
	inten = max(infile.cluster_id)


	final = {'type':'FeatureCollection', 'features': []}

	for i in range(1,inten+1):
		arr =[]
		elev=max(infile.height[infile.cluster_id==i])
		elev=float(elev)

		clusterx = infile.x[infile.cluster_id ==i]
		clustery = infile.y[infile.cluster_id ==i]

		points =np.vstack([clusterx,clustery]).T

		hull = ConvexHull(points)
		for hv in hull.vertices:
			arr.append([points[hv,0],points[hv,1]])
		polygon=Polygon(arr)
		point=polygon.centroid
		coord=list(polygon.exterior.coords)
		r=10000
		for c in range(1,len(coord)):
			line=LineString([coord[c],coord[c-1]])
			d=line.distance(point)
			if d<r:
				r=d

		feature ={"type":"Feature","geometry":{"type":"Point","coordinates":[point.x,point.y]},"properties":{"id":int(i),"height":float(elev),"radius":float(r),"area":float(math.pi*r*r)}}
		final['features'].append(feature)


	with open('./data/interim/Trees_data_'+filename[:-4]+'.json', 'w') as outfile:
		json.dump(final, outfile)

def Mergingpolygons(filename):
	# reading into two geojson objects, in a GCS (WGS84)
	Heights=[]
	Radius=[]
	with open('./data/interim/Trees_data_'+filename[:-4]+'.json') as geojson1:
		poly1_geojson = json.load(geojson1)
		x=poly1_geojson['features']
		for t in x:
			h=t['properties']
			Heights.append(h['height'])
			Radius.append(h['radius'])
	poly=[]


	for i in range(len(poly1_geojson['features'])):
		poly.append(shapely.geometry.asShape(poly1_geojson['features'][i]['geometry']))
	index=list(range(0,len(poly)))

	# checking to make sure they registered as polygons
	count=1
	while(count<11):
		index=list(range(0,len(poly)))
		for i in index:
			for j in index:
				if (len(poly)>i) and (len(poly)>j):
					if (poly[i].distance(poly[j])<3.75  and i!=j ):
						x=(poly[i].x+poly[j].x)/2
						y=(poly[i].y+poly[j].y)/2
						d=poly[i].distance(poly[j])
						if Radius[i]>=d+Radius[j]:
							del poly[j]
							del Heights[j]
							del Radius[j]
						elif Radius[j]>=d+Radius[i]:
							poly[i]=poly[j]
							Heights[i]=Heights[j]
							Radius[i]=Radius[j]
							del poly[j]
							del Heights[j]
							del Radius[j]
						else:
							x=(((Radius[i]-Radius[j]+d)*poly[i].x)+((Radius[j]-Radius[i]+d)*poly[j].x))/(2*d)
							y=(((Radius[i]-Radius[j]+d)*poly[i].y)+((Radius[j]-Radius[i]+d)*poly[j].y))/(2*d)
							Radius[i]=(d+Radius[i]+Radius[j])/2
							poly[i]=Point(x,y)
							Heights[i]=max(Heights[i],Heights[j])
							del poly[j]
							del Heights[j]
							del Radius[j]
						index.remove(j)
						for n, k in enumerate(index):
							if k>j:
								index[n] = index[n]-1
		count=count+1



	final = {'type':'FeatureCollection', 'features': []}

	for i in range(len(poly)):
			geojson_out=geojson.Feature(geometry=poly[i])
			feature ={"type":"Feature","geometry":{"type":"Polygon","coordinates":[]},"properties":{"id":i+1,"height":Heights[i],"radius":Radius[i],"area":Radius[i]*Radius[i]*math.pi}}
			feature['geometry']=geojson_out.geometry
			final['features'].append(feature)

	with open('./data/external/Trees_'+filename[:-4]+'.json', 'w') as outfile:
		json.dump(final, outfile)


