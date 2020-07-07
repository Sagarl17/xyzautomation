import os
import sys
import math
import json
import laspy
import shapely
import geojson
import numpy as np
import pandas as ps
from shapely.geometry import Point

filename=sys.argv[1]

filepath='data/raw/'+filename


#Determine the size of the file and the number of pieces to divide the file to
file_stats=os.stat(filepath)
file_size=file_stats.st_size/(1024*1024*1024)
pieces_count=int(file_size//1+1)

#Opening las file and storing the points in numpy array
infile=laspy.file.File(filepath)
col = {'x':infile.x, 'y':infile.y, 'z':infile.z, 'r':infile.red/256, 'g':infile.green/256, 'b':infile.blue/256, 'c':infile.classification}
data = ps.DataFrame(data=col)
data=data[['x', 'y', 'z', 'r', 'g', 'b', 'c']].to_numpy()

#Get the range of x and y coordinates to determine the direction for dividing las file
max_x=np.amax(data[:,0])
min_x=np.amin(data[:,0])

max_y=np.amax(data[:,1])
min_y=np.amin(data[:,1])

#Get the range value and the dimension
if max_x-min_x> max_y-min_y:
    max_c=max_x
    min_c=min_x
    dimension='x'
else:
    max_c=max_y
    min_c=min_y
    dimension='y'

dif_c=(max_c-min_c)/pieces_count

#create empty array to store the names of th created point clouds
pointclouds=[]



count=1
while abs(max_c - min_c) >1:                                                                            #Create while loop while accounting for decimal errors
    print(max_c,min_c)
    if dimension=='x':
        x,index=np.where([(data[:,0]>= min_c) & (data[:,0]<min_c+dif_c)])                               #Get the index of the points which lie in the range specified
    else:
        y,index=np.where([(data[:,1]>= min_c) & (data[:,1]<min_c+dif_c)])
    
    point_to_store = np.take(infile.points,index)                                                       #Extract the points and store it in new las file
    outfile=laspy.file.File("data/raw/"+filename[:-4]+str(count)+'.las',mode="w",header=infile.header)
    outfile.points=point_to_store
    outfile.close()
    pointclouds.append(filename[:-4]+str(count)+'.las')
    count=count+1
    min_c=min_c+dif_c

for cloud in pointclouds:
    os.system('python3 main.py '+cloud+' trees')                                                         #Run the command calling it within the script

jsons=os.listdir('data/external/')

features=[]




for pc in pointclouds:
    for j in jsons:
        if pc[:-4] in j:
            final=json.load(open('data/external/'+j))
            features=features+final['features']

final = {'type':'FeatureCollection', 'features': []}
final['features']=features
with open('./data/external/Trees_'+filename[:-4]+'.json', 'w') as outfile:
		json.dump(final, outfile)


# reading into two geojson objects, in a GCS (WGS84)
Heights=[]
Radius=[]
with open('./data/external/Trees_'+filename[:-4]+'.json') as geojson1:
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