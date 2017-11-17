# wfc3_photometry

Welcome to the WFC3 Photometric tools repository.  This repository will be used as the home base for all the tools we develop to unify/optimize our photometry data analysis.  For now, the goal of this project is to create a unified pipeline that encompasses the work done by several team members, to minimize the amount of repeated efforts/errors in processes.  Specifically we want to develop a comprehensive pipeline for analyzing the photometric standard star images, but the tools in this repository will probably work for other data.

The pipeline will be run via the use of config files.  The config files will be stored with the outputs,

### Rough skeleton of pipeline:
1. Query MAST Database for all observations of target(s)
    1. Check calibration version?
2. Download all observations not in data directories (Via Astroquery or QL?)
3. Filter bad datasets
4. Visit level drizzle all images
5. Detect source in each drizzled image?
    1. Use WCS tools to transform coordinates to find them in FLT/FLC frame
    2. Validate detections?
6. Run LACosmic on relevant exposures, save output images in a parallel tree.
7. Run the photometry, use several different apertures, bg methods etc.
    1. Potentially save photometric results separately from bg results?
8. Save results to csv's?  hdf5?  Some format that will be easy for anyone to access
9. Make some preliminary plots from these results.


### Ideas:
* Base the flow of the software on the contam pipeline?
* Write as little new code as possible, use packages that are well developed and maintained
* Document the pipelines well so others can work on the project without a huge learning curve
