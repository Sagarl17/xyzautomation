import sys
import json

import math
import fiona
import laspy
import numpy as np
import multiprocessing
from shapely import geometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon ,LineString,Point

filename=sys.argv[1]


class featurecalculation:
    def features(self):
        farr=[]
        pool = multiprocessing.Pool(processes=division)     # Create a multiprocessing Poolfor div in range(division):
        result=pool.map(self.calc, range(division),chunksize=1)  # process data_inputs iterable with pool
        for divo in range(division):
            farr=farr+result[divo]

        final['features']=farr
        with open('./data/interim/Trees_data_'+filename[:-4]+'.json', 'w') as outfile:
            json.dump(final, outfile)
    
    
    def calc(self,div):
        arr2=[]
        for t in range(div*maximum_points,min((div+1)*maximum_points,inten)):
            arr =[]
            elev=max(infile.height[infile.cluster_id==t])
            elev=float(elev)
            clusterx = infile.x[infile.cluster_id ==t]
            clustery = infile.y[infile.cluster_id ==t]

            points =np.vstack([clusterx,clustery]).T
            
            hull = ConvexHull(points)
            for i in hull.vertices:
                arr.append([points[i,0],points[i,1]])
            polygon=Polygon(arr)
            point=polygon.centroid
            coord=list(polygon.exterior.coords)
            r=10000
            for c in range(1,len(coord)):
                line=LineString([coord[c],coord[c-1]])
                d=line.distance(point)
                if d<r:
                    r=d
            
            feature ={"type":"Feature","geometry":{"type":"Point","coordinates":[point.x,point.y]},"properties":{"id":t,"height":elev,"radius":r,"area":math.pi*r*r}}
            arr2.append(feature)
        return arr2

infile = laspy.file.File('./data/processed/Trees_Clustered_'+filename,mode ='rw')
inten = max(infile.cluster_id)
division=multiprocessing.cpu_count()
final = {"type":"FeatureCollection", "features": []}
maximum_points=inten//division+1
fe=featurecalculation()
fe.features()
