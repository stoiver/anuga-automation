# ANUGA Hydrodynamic Modeling: Hands On Exercises for flood inundation modeling

# Author information:
Trushnamayee Nanda

### Notebook Series Description:
The series of three notebooks show usage of a unstructured mesh generation, model&mesh setup and run model, respectively.  
### 1) Mesher.ipynb
### 2) Setup_model.py #
### 3) Run_model.py
Note: `config_example.yaml` file contains all hardcoded parameters defined in it


#Necessary Datasets:

Please ensure that all the following datasets were downloaded and placed into the "data" subfolder. Zip files should be unziped in the same folder, 

## DEM.tif, river_Centerline.shp, upstream_boundary_line.shp and downstream_boundary_line.shp

1) Mesher.ipynb contains python script to be used in a conda environment where mesher is pre-installed.(run Mesher.ipynb in jupyter notebook for generating random noise raster around river buffer and subsequently to generate triangular UnStructuredMesh)

It can be used with the DEM.tif and any constraint file such as riverline, flow accumulation, random noise raster etc.
This code will generate an unstructured mesh file name *_USM.shp in the folder specified in the code. After the *_USM.shp is generated, copy the .shp and supporting files into "data" directory 

2) Setup_model.py contains script to read (through config file) DEM and boundary_line files, reading mesh triangles (sanity check for redundant, very small area and other degenerative triangles), extract triangle points and vertices, and interpolating the DEM at triangle vertices 

3)  Run_model.py makes the user do the following steps (using config.yaml and outputs of setup_model)
- uses files generated from mesh such as pts_npy, tris_npy
- sanitizes triangles and makes CCW
- builds domain (as per ANUGA function)
- assigns boundary tags (as per ANUGA function)
- finds inlet, outlet and exterior (as per ANUGA function)
- assigns inlet operator for input discharge (as per ANUGA function)


### How to run the model without hardcoded parameter values, user can customize in config_example.yaml file###

mpiexec -n 4 python run_model.py --config config_example.yaml

Post-processing

###use of merge_sww.py in case files are not merged at the end of run_model
  python merge_sww.py

###use of anuga_output_analysis.py for getting animation for .sww file
  python anuga_output_analysis.py --sww 20140702000000_Shellmouth_flood_12_days.sww --make-video depth --fps 20 --clim-depth 0,10

###use of export_depth_wse_velocity.py for generating .tif raster for inundation depth, wse, velocity eg., extract results for 09-07-2014 00UTC
   python export_depth_wse_velocity.py --sww 20140702000000_Shellmouth_flood_12_days.sww --dem-mask "./data/DEM_MTI_PART.tif" --outdir ./exports_inundation --start "2014-07-02 00:00:00"--dates "2014-07-09 00:00:00" --res 1 --epsg 26914 --depth-min 0.3
   

