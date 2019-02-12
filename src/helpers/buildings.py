import os
import laspy
import numpy as np
from scipy.spatial import KDTree,ConvexHull
from shapely import geometry
import json
import shapely.geometry
import shapely.ops
import geojson
import progressbar

def building_LAS(ground_file_name):
	infile=laspy.file.File('./'+ground_file_name,mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z,infile.red,infile.green,infile.blue]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if classess[i]==6]
	point_to_store = np.take(infile.points,cand)
	point_to_return = point_3d[cand]
	outfile_name = "buildings.las"
	outfile=laspy.file.File("./data/processed/"+outfile_name,mode="w",header=infile.header)
	outfile.points=point_to_store
	outfile.close()

	return point_to_return,infile.header

def Clustering():
	infile=laspy.file.File('./data/processed/buildings.las',mode='rw')
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
			for point in Q:
				NP=tree.query_ball_point(point_3d[point],r = 0.175)
				for point1 in NP:
					l1=len(FC)
					FC.add(point1)
					l2=len(FC)
					if l2>l1:
						Q.append(point1)
						bar.update(len(FC))
			F=[]					
			if len(Q)>2500:
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
	outfile=laspy.file.File("Best.las",mode="w",header=infile.header)
	outfile.points=points
	outfile.close()
	infile=laspy.file.File("Best.las",mode="r")

	header = infile.header
	outfile=laspy.file.File('./data/processed/Buildings_Clustered.las',mode="w",header=header)
	outfile.define_new_dimension(name = "cluster_id",data_type = 5, description = "Cluster_id")
	outfile.cluster_id=intensity
	print(len(infile.points))
	outfile.x = infile.x
	outfile.y = infile.y
	outfile.z = infile.z
	outfile.red = infile.red
	outfile.green = infile.green
	outfile.blue = infile.blue
	bar.finish()
	outfile.close()
	os.remove("Best.las")
	return

def Polygonextraction():
	infile = laspy.file.File('./data/processed/Buildings_Clustered.las',mode ='rw')
	arr =[]
	inten = max(infile.cluster_id)

	final = {"type":"FeatureCollection", "features": []}

	for i in range(1,inten+1):
		print(i)
		arr =[]
		feature ={"type":"Feature","geometry":{"type":"Polygon","coordinates":[]},"properties":{"id":i}}
		clusterx = infile.x[infile.cluster_id ==i]
		clustery = infile.y[infile.cluster_id ==i]

		clusterz = infile.z[infile.cluster_id ==i]

		clusterz.sort()
		height = (sum(clusterz[-101:-1]) - sum(clusterz[0:100]))/100
		maxh=clusterz[-1]-clusterz[0]


		points =np.vstack([clusterx,clustery]).T
		points=points.tolist()
		p=set()
		for i in points:
			p.add((i[0],i[1]))
		points=list(p)
		division=len(points)//250+1
		poly=[]
		for div in range(division):
			small_points = points[div*250:(div+1)*250]
			if len(small_points)>10:
				sp=[]
				hull=ConvexHull(small_points)
				for i in hull.vertices:
					sp.append(small_points[i])
				x=geometry.Polygon(sp)
				poly.append(x)
		count=1
		while(count<11):
			index=list(range(0,len(poly)))
			for i in index:
				for j in index:
					if i>=len(poly) or j>=len(poly):
						break
					elif (poly[i].intersects(poly[j])==True and i!=j ):
						poly[i]=poly[i].union(poly[j])
						del poly[j]
						index.remove(j)
						for n, k in enumerate(index):
							if k>=j:
								index[n] = index[n]-1
			count=count+1
		x, y = poly[0].exterior.coords.xy
		for i in range(len(x)):
			arr.append([x[i],y[i]])

		feature['geometry']['coordinates'].append(arr)
		feature['properties']['height'] = maxh
		final['features'].append(feature)


	with open('./data/interim/Buildings_data.json', 'w') as outfile:
		json.dump(final, outfile)

def Mergingpolygons():
	# reading into two geojson objects, in a GCS (WGS84)
	Heights=[]
	with open('./data/interim/Buildings_data.json') as geojson1:
		poly1_geojson = json.load(geojson1)
		x=poly1_geojson['features']
		for t in x:
			h=t['properties']
			Heights.append(h['height'])
	poly=[]


	for i in range(len(poly1_geojson['features'])):
		poly.append(shapely.geometry.asShape(poly1_geojson['features'][i]['geometry']))

	count=1
	while(count<11):
		print(count*10)
		index=list(range(0,len(poly)))
		# checking to make sure they registered as polygons
		for i in index:
			for j in index:
				if (len(poly)>i) and (len(poly)>j):
					print(len(poly),i,j)
					if (poly[i].intersects(poly[j])==True and i!=j ):
						if ((1-(poly[i].difference(poly[j]).area)/poly[i].area)>0.25  or  poly[i].contains(poly[j])==True ) :
							poly[i]=poly[i].union(poly[j])
							del poly[j]
							del Heights[j]
							index.remove(j)
							for n, k in enumerate(index):
								if k>j:
									index[n] = index[n]-1
						elif ((1-(poly[j].difference(poly[i]).area)/poly[j].area)>0.25  or  poly[j].contains(poly[i])==True ) :
							poly[j]=poly[i].union(poly[j])
							del poly[i]
							del Heights[i]
							index.remove(i)
							for n, k in enumerate(index):
								if k>i:
									index[n] = index[n]-1
		count=count+1


	final = {"type":"FeatureCollection", "features": []}

	for i in range(len(poly)):
			geojson_out=geojson.Feature(geometry=poly[i])
			feature ={"type":"Feature","geometry":{"type":"Polygon","coordinates":[]},"properties":{"id":i,"Height":Heights[i]}}
			feature['geometry']=geojson_out.geometry
			final['features'].append(feature)


	with open('./data/external/Buildings.json', 'w') as outfile:
	    json.dump(final, outfile)
	return
