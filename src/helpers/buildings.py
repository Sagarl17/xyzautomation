import laspy
import numpy as np
from scipy.spatial import KDTree,ConvexHull
import json
import shapely.geometry
import shapely.ops
import geojson

def building_LAS(ground_file_name):
	infile=laspy.file.File('./'+ground_file_name,mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z,infile.red,infile.green,infile.blue]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if ((classess[i]==6 and ((point_3d[i,4]<0.45*point_3d[i,3]+0.61*point_3d[i,5])and ((.85*point_3d[i,4]<point_3d[i,3])and(.7*point_3d[i,4]<point_3d[i,5])))) and point_3d[i,2]>0)]
	point_to_store = np.take(infile.points,cand)
	point_to_return = point_3d[cand]
	outfile_name = "buildings.las"
	outfile=laspy.file.File("./data/processed/"+outfile_name,mode="w",header=infile.header)
	outfile.points=point_to_store
	outfile.close()

	return point_to_return,infile.header

def Clustering():
	infile=laspy.file.File('./data/processed/buildings.las',mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z]).T
	d=len(point_3d)

	tree = KDTree(point_3d)
	points=[]
	intensity=[]
	FC=set()                                               # Intialize empty cluster list
	Q=[]
	i=1
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
						print((l2/d)*100)
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
				print(i)
				i=i+1
				F=F+Q
			Q=[]
	outfile=laspy.file.File("./data/interim/Clustered.las",mode="w",header=infile.header)
	outfile.points=points
	outfile.intensity=intensity
	outfile.close()
	return

def Polygonextraction():
	infile = laspy.file.File('./data/interim/Clustered.las',mode ='rw')
	inten = max(infile.intensity)
	final = {"type":"FeatureCollection", "features": []}
	for i in range(1,inten+1):
		arr =[]
		feature ={"type":"Feature","geometry":{"type":"Polygon","coordinates":[]},"properties":{"id":i}}
		print((i/inten)*100)
		clusterx = infile.x[infile.intensity ==i]
		clustery = infile.y[infile.intensity ==i]
		points =np.vstack([clusterx,clustery]).T
		hull = ConvexHull(points)
		for i in hull.vertices:
			arr.append([points[i,0],points[i,1]])
		feature['geometry']['coordinates'].append(arr)
		final['features'].append(feature)

	with open('./data/interim/buildings.json', 'w') as outfile:
		json.dump(final, outfile)
	return

def Mergingpolygons():
	# reading into two geojson objects, in a GCS (WGS84)
	with open('./data/interim/buildings.json') as geojson1:
		poly1_geojson = json.load(geojson1)
	poly=[]


	for i in range(len(poly1_geojson['features'])):
	    poly.append(shapely.geometry.asShape(poly1_geojson['features'][i]['geometry']))
	index=list(range(0,len(poly)))
	# checking to make sure they registered as polygons
	for i in index:
	    for j in index:
	        if (poly[i].intersects(poly[j])==True and i!=j ):
	            if ((1-(poly[i].difference(poly[j]).area)/poly[i].area)>0.25  or  poly[i].contains(poly[j])==True ) :
	                poly[i]=poly[i].union(poly[j])
	                del poly[j]
	                index.remove(j)
	                for n, k in enumerate(index):
	                    if k>j:
	                        index[n] = index[n]-1
	            elif ((1-(poly[j].difference(poly[i]).area)/poly[j].area)>0.25  or  poly[j].contains(poly[i])==True ) :
	                poly[j]=poly[i].union(poly[j])
	                del poly[i]
	                index.remove(i)
	                for n, k in enumerate(index):
	                    if k>i:
	                        index[n] = index[n]-1


	final = {"type":"FeatureCollection", "features": []}

	for i in range(len(poly)):
	        geojson_out=geojson.Feature(geometry=poly[i])
	        feature ={"type":"Feature","geometry":{"type":"Polygon","coordinates":[]},"properties":{"id":i}}
	        feature['geometry']=geojson_out.geometry
	        final['features'].append(feature)

	with open('./data/external/buildings.json', 'w') as outfile:
	    json.dump(final, outfile)
	return
