for running the code python main.py <LAS file directory>


the program will create several intermediate data so, ignore the other data expect :

1. classified_las_file.las :- Classified LAS file for segmentation in step1.
2. tree_data.json :- Contains Tree data like location,radius,height and geo coordinates (lat-long).
3. planes_2D_500.shx/shp/dbf :- Shapefile for the roof top of the building.
4. plane_shapes.shx/shp/dbf :- Shapefile for different planes existing in buildings in whole las file.
5. planes_data.json :- Planes Parameter such as lower height , upper height and 3D polygon vertices.
6. planes_CAD.off :- Object file which created for visulaization of building with a water tight model (can be opened in MeshLab).


Some Timing Analysis:

For Point Cloud Classification :- 10 minutes / 1 Million Points.
For Trees Extraction :- 1.5 minutes / 1 Million Trees Points. 
For Roof Top Extraction :- 1.5 minutes / 1 Million Building Points.
For Building Planes Extraction :- 24 minutes / 1 Million Building Points.
For Water Tight Building Model :- 1 minutes atmost (instanteneous though)
