# apparitionAgriculture

This repo contains the basis for the data processing and structuration for running models on the birth of agriculture.

## Data processoing code

### extract_graph.ipynb
This notebook contains the data processing to extract neighboring nodes in the hexagonal partitionning of the world map.

### plot_graph.ipynb
This notebook contains the data checking and creation of neighbors graph to check the consistency of the graph created.

### hexagon_ray.ipynb
This notebook checks the ray of an hexagon in terms of latitude and longitude, in order to know which data is in which hexagon.

### data_to_graph.py
This script extracts the weather data and maps it on our hexagonal partition of the world. It also generates some plots to visualize the data.

### data_to_graph2.py
This script extracts the biodiversity data and maps it on our hexagonal partition of the world. It also generates some plots to visualize the data.

## Data generated for the models

### world.csv
This file contains the data about the neighboring hexagones of each heaxagon, labeled from 0 to 5 depending on the location of the neighbor, according to the following labeling. It is generated in the notebook extract_graph.ipynb. The process is quite data-heavy so we use a multiprocessing module (dask) to parallelize the computations.

      0
    _____
 5 /     \ 1
  /       \
  \       /
 4 \_____/ 2
      3

### world_coord.csv
This file contains the coordinates of the barycenter of each hexagon. It is generated in the notebook extract_graph.ipynb.

### world_data_weather.csv
This file contains the data generated by data_to_graph.py after mapping the raw climate data on our hexagonal partitionning of the world.

### world_data_biodiversity.csv
This file contains the data generated by data_to_graph2.py after mapping the raw biodiversity data on our hexagonal partitionning of the world.

## Running the model

To be continued ...