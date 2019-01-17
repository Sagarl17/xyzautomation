import laspy
import numpy as np
from scipy.spatial import KDTree,ConvexHull
import json
from shapely.geometry import Polygon,mapping
from collections import defaultdict,OrderedDict
from pyproj import Proj,transform
from scipy import spatial,optimize

def points_from_LAS(input_file_name):
	'''
	extracting the tree points from the classified point cloud
	'''
	infile=laspy.file.File('./'+input_file_name,mode='rw')
	point_3d=np.vstack([infile.x,infile.y,infile.z]).T
	classess=infile.classification
	cand = [i for i in range(len(point_3d)) if classess[i]==5]
	point_3d = np.take(infile.points,cand)

	outfile_name = "full_trees.las"
	outfile=laspy.file.File("./data/processed/"+outfile_name,mode="w",header=infile.header)
	outfile.points=point_3d
	outfile.close()
	infile.close()
	return outfile_name

def tree_top_cand(trees_file):
	'''
	Finding the Tree Tops as Seed for CLustering Based on Local Maximums
	'''
	distance_b = 300
	infile = laspy.file.File("./data/processed/"+trees_file,mode='rw')
	point_3d = np.vstack([infile.x,infile.y,infile.z]).T
	scales = infile.header.scale
	offsets = infile.header.offset
	count_tree_tops = np.shape(point_3d)[0]//distance_b + 1
	int_tree_t = []
	for i in range(count_tree_tops):
		pts = point_3d[i*distance_b:(i+1)*distance_b]
		z_c = pts[:,2]
		local_max = np.argmax(z_c)
		int_tree_t.append(pts[local_max])
	infile.close()
	return int_tree_t,count_tree_tops,point_3d,scales,offsets

def dfs(adj_list, visited, vertex, result, key):
    visited.add(vertex)
    result[key].append(vertex)
    for neighbor in adj_list[vertex]:
        if neighbor not in visited:
            dfs(adj_list, visited, neighbor, result, key)

def merging_adj_ttops(all_tree_top,no_intial_ttops):
	'''
	Merging the Tree Tops which are very close to each other
	'''
	merging_ttop_dist = 5
	tree_ttops = KDTree(np.asarray(all_tree_top)[:,0:2])
	b = tree_ttops.query_pairs(merging_ttop_dist)
	adj_list = defaultdict(list)
	for x, y in b:
		adj_list[x].append(y)
		adj_list[y].append(x)

	result = defaultdict(list)
	visited = set()
	for vertex in adj_list:
		if vertex not in visited:
			dfs(adj_list, visited, vertex, result, vertex)

	all_train = []
	for i in result.values():
		for j in i:
			all_train.append(j)
	a = np.array([x for x in range(no_intial_ttops)])
	remain = np.in1d(a,all_train,invert=True)
	remaining_clusters = a[remain]
	new_int_tree = []
	for j in remaining_clusters:
		new_int_tree.append(all_tree_top[j])
	for i in result.values():
		new_int_tree.append((np.asarray(all_tree_top)[i])[np.argmax((np.asarray(all_tree_top)[i])[:,2])])

	return new_int_tree

def getting_neighbour(new_ttops,point_3d):
	'''
	Clustering Based on the new tree tops and since there is a
	correlation between height of the tree and radius of the tree using
	the clustering threshold in terms of height only
	'''
	clustering_threshold_radius = 0.6
	tree = KDTree(point_3d[:,0:2])
	neighbours = []
	point_completed = []
	for tree_top in new_ttops:
		pt_2d = tree_top[:2]
		height = tree_top[2]
		radius = height*clustering_threshold_radius
		neighbour = np.setdiff1d(tree.query_ball_point(pt_2d,radius),point_completed,assume_unique=True)
		neighbours.append(neighbour)
		point_completed.extend(neighbour)
	return neighbours

def color_tree_las(neighbours,point_3d):
	'''
	Saving the LAS file for better visulaization of different trees
	'''
	color_dictionary = [[192,192,255],[192,192,128],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[192,192,192]
						,[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],[0,128,128],[0,0,128]]
	colors = np.shape(neighbours)[0]

	color_array = []
	for le in range(colors):
		color_array.append(color_dictionary[le%(len(color_dictionary))])

	X_coor,Y_coor,Z_coor = [],[],[]
	r,g,b = [],[],[]
	for i in range(len(neighbours)):
		if np.shape(point_3d[neighbours[i]])[0] > 100:
			array_data = np.asarray(point_3d[neighbours[i]],dtype="int")
			X_coor = np.concatenate((np.asarray(X_coor),np.asarray(array_data[:,0])),axis = 0)
			Y_coor = np.concatenate((np.asarray(Y_coor),np.asarray(array_data[:,1])),axis = 0)
			Z_coor = np.concatenate((np.asarray(Z_coor),np.asarray(array_data[:,2])),axis = 0)
			r = np.concatenate((np.asarray(r),np.full(np.shape(array_data)[0], color_array[i][0])),axis = 0)
			g = np.concatenate((np.asarray(g),np.full(np.shape(array_data)[0], color_array[i][1])),axis = 0)
			b = np.concatenate((np.asarray(b),np.full(np.shape(array_data)[0], color_array[i][2])),axis = 0)

	outfile=laspy.file.File("./data/interim/trees_colored.las",mode="w", header=Header_for_all)
	outfile.X=X_coor
	outfile.Y=Y_coor
	outfile.Z=Z_coor
	outfile.red = np.array(r)
	outfile.green = np.array(g)
	outfile.blue = np.array(b)
	outfile.close()

def tree_parameters_npy(neighbours,point_3d,scales,offsets):
	'''
	Saving the Trees Parameter in different Formats as Required
	'''
	polygons = []
	height_t = []
	for i in range(len(neighbours)):
		if np.shape(neighbours[i])[0] != 0 and np.shape(point_3d[neighbours[i]])[0]>10:
			array = point_3d[neighbours[i]]
			array = np.asarray(array)
			i_2d = array[:,0:2]
			ht_tree = np.max(array[:,2])
			hull = ConvexHull(i_2d)
			polygons.append(array[hull.vertices])
			height_t.append(ht_tree*scales[2]+offsets[2])
	all_data = np.array([polygons,height_t])
	all_data = np.asarray(all_data)
	#out_np_file = "tree_parameters"
	#np.save(out_np_file , all_data)
	return all_data

def trees_parameter_json(parameters_tree,radius,location):
    features = []
    for i in range(np.shape(parameters_tree[1])[0]):
        data_dict = OrderedDict()
        data_dict["type"] = "Feature"
        new_dict = OrderedDict()
        new_dict["TreeHeight"] = float(parameters_tree[1][i])
        new_dict["TreeRadius"] = float(radius[i])
        data_dict["properties"]=  new_dict
        data_dict["geometry"] = OrderedDict(type = "Point",
                                    coordinates = location[i])
        features.append(data_dict)

    data = OrderedDict()
    data["type"] = "FeatureCollection"
    data["crs"] = OrderedDict(type = "name",properties = {"name":"urn:ogc:def:crs:OGC:1.3:CRS84"})
    data["features"] = features
    with open('./data/external/tree_data.json', 'w') as f:
        json.dump(data,f)

def shp_for_2D_polygon(polygons):

	''' finding the shapefile for each tree '''
	multipoly_3D=[]
	for i in polygons:
		poly = Polygon(i)
		multipoly_3D.append(poly)

	w3D = shapefile.Writer(shapeType=shapefile.POLYGON)
	count_shp=0
	w3D.field("ID")
	w3D.autoBalance = 1
	for polygon in multipoly_3D:
		count_shp=count_shp+1
		polypar = []
		polygon = mapping(polygon)["coordinates"][0]
		for i in polygon:
			fro = []
			for dummy in i:
				fro.append(float(dummy))
			polypar.append(fro)
		w3D.poly(parts=[polypar])
		w3D.record(count_shp)
	w3D.save('polygons')

#############################################################################
# 				Converting to LAT-LONG-Height coordinates                   #

def Proj_to_latlong(polygons,scales,offsets):

	inProj = Proj(init='epsg:32644')
	outProj = Proj(init='epsg:4326')
	location = []
	radius = []
	for i in polygons:
		r,c = leastsq_circle(i[:,0]*scales[0]+offsets[0],i[:,1]*scales[1]+offsets[1],i[:,2]*scales[2]+offsets[2])
		x,y = transform(inProj,outProj,c[0],c[1])
		cent = [x,y,c[2]]
		location.append(cent)
		radius.append(r)

	return radius,location

def leastsq_circle(x,y,z):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    z_m = np.mean(z)
    center_estimate = x_m, y_m, z_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y,z))
    xc, yc,zc = center
    Ri       = calc_R(x, y,z, *center)
    R        = Ri.mean()
    return R,center



def calc_R(x,y,z, xc, yc, zc):
    return np.sqrt((x-xc)**2 + (y-yc)**2 + (z-zc)**2)

def f(c, x, y,z):
    Ri = calc_R(x, y,z, *c)
    return Ri - Ri.mean()
