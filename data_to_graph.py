from scipy.spatial import KDTree
import numpy as np
import csv
import pandas as pd
import functools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

graph = {}
with open('world_coord.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        if len(rows[0]):
            graph[int(rows[0])]=(float(rows[1]),float(rows[2]))

raw_data = {}
raw_data["elevation"] = np.load("data/elevation.npy")
raw_data["annual_pp"] = np.load("data/annual_pp.npy")
raw_data["max_pp"] = np.load("data/max_pp.npy")
raw_data["min_pp"] = np.load("data/min_pp.npy")
raw_data["pp_seasonality"] = np.load("data/pp_seasonality.npy")
raw_data["min_temp"] = np.load("data/min_temp.npy")
raw_data["max_temp"] = np.load("data/max_temp.npy")
raw_data["temp_seasonality"] = np.load("data/temp_seasonality.npy")

data={}
for key in raw_data:
    for row in raw_data[key]:
        id=(row[1],row[2])
        if id not in data:
            data[id]={}
        if row[0]>-1000:
            data[id][key]=row[0]

print(len(data))

for key in list(data.keys()):
    if len(data[key])<8:
        del(data[key])

## Possible improvement : one kd tree per data, drop only data nan per column and not if one of the fields is nan

print(len(data))

data_points = list(data.keys())
wanted_ids = list(graph.keys())
wanted_points = [graph[i] for i in wanted_ids]

kdtree = KDTree(data_points)

# Query for the closest neighbors
nb_neighbors=len(data)//len(wanted_ids)
print(nb_neighbors)
distances, neighbors_indices = kdtree.query(wanted_points, k=nb_neighbors)

# print(neighbors_indices)

# Get the values of the closest neighbors
closest_values = {}
for key in raw_data.keys():
    closest_values[key] = [[data[data_points[j]][key] for j in neighbors_indices[i]] for i in range(len(wanted_points))]

# Calculate the mean of the values
mean_values = {}
for key in raw_data.keys():
    mean_values[key] = np.mean(closest_values[key], axis=1)

print(mean_values["elevation"].shape)

df = pd.DataFrame(mean_values, index=wanted_ids)
print(df)


x,y,z=[],[],[]
for i in range(len(mean_values["min_temp"])):
    key=wanted_points[i]
    val=mean_values["min_temp"][i]
    x.append(key[0])
    y.append(key[1])
    z.append(val)

fig = plt.figure()
ax = fig.add_subplot()
sct = ax.scatter(x, y, c=z, marker='o', cmap=cm.copper, s=1)
ax.set_title('Figure 1. Geodistribution of minimal temperature')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.colorbar(sct, orientation="vertical")
plt.show()

x,y,z=[],[],[]
for i in range(len(mean_values["pp_seasonality"])):
    key=wanted_points[i]
    val=mean_values["pp_seasonality"][i]
    x.append(key[0])
    y.append(key[1])
    z.append(val)

fig = plt.figure()
ax = fig.add_subplot()
sct = ax.scatter(x, y, c=z, marker='o', cmap=cm.copper, s=1)
ax.set_title('Figure 1. Geodistribution of precipitation seasonality')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.colorbar(sct, orientation="vertical")
plt.show()


x,y,z=[],[],[]
for i in range(len(mean_values["max_temp"])):
    key=wanted_points[i]
    val=mean_values["max_temp"][i]
    x.append(key[0])
    y.append(key[1])
    z.append(val)

fig = plt.figure()
ax = fig.add_subplot()
sct = ax.scatter(x, y, c=z, marker='o', cmap=cm.copper, s=1)
ax.set_title('Figure 1. Geodistribution of maximal temperature')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.colorbar(sct, orientation="vertical")
plt.show()

df.to_csv("world_data_weather.csv")
