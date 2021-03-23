##### CSV files #####

The historical_sample.csv is the result of the get_master_data jupyter notebook.
The map.csv contains only distkm (distance in km), solar_power (solar power), and elevation angle, bearing and speed limit.

##### Notebooks #####

rl_solar_ray: to run training, and validation
get_master_data: to filter data into historical_sample.csv

##### python files #####

utils: contains helper functions

##### Other files and folders #####

Dockerfile
src folder containing environment and training. 
common folder contains ray dependencies
