import datetime
import time
import psutil
import numpy as np
from scipy import spatial,optimize
from pyproj import Proj,transform


def print_time():
	""" For printing DATETIME """
	return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def memcalc():
	mem='RAM: '+str(psutil.virtual_memory()[2])+'%'
	return mem

def cpucalc():
	cpu='CPU: '+str(psutil.cpu_percent(interval=None, percpu=False))+'%'
	return cpu


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calc_R(x,y,z, xc, yc, zc):
    """ Finding Radius from polygons and location as center for the trees """
    return np.sqrt((x-xc)**2 + (y-yc)**2 + (z-zc)**2)

def f(c, x, y,z):
    Ri = calc_R(x, y,z, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y,z):
    """ coordinates of the barycenter """
    x_m = np.mean(x)
    y_m = np.mean(y)
    z_m = np.mean(z)
    center_estimate = x_m, y_m, z_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y,z))
    xc, yc,zc = center
    Ri       = calc_R(x, y,z, *center)
    R        = Ri.mean()
    return R,center

def Proj_to_latlong(polygons,scales,offsets):
	""" Converting UTM to LAT-LONG-Height coordinates """
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
