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
raw_data["animals"] = np.load("data/animalBiodiversity.npy")
raw_data["plants"] = np.load("data/wildCropsCousinsDiversity.npy")

method = {"animals" : "mean", "plants" : "sum"}

for key in raw_data:
    if len(raw_data[key].shape) > 2:
        t=raw_data[key].shape
        new_shape=(np.prod(t[:-1]),t[-1])
        raw_data[key]=raw_data[key].reshape(new_shape)
    print(raw_data[key].shape)

keys=list(raw_data.keys())
data=[]
for key in raw_data:
    data.append({})
    for row in raw_data[key]:
        if key=="plants":
            id=(row[2],row[1])
        else:
            id=(row[1],row[2])
        data[-1][id]=row[0]

print(len(data[0]))

data_points_list = [list(sub_data.keys()) for sub_data in data]
wanted_ids = list(graph.keys())
wanted_points = [graph[i] for i in wanted_ids]

result={}
# Query for the closest neighbors
for k,data_points in enumerate(data_points_list):
    kdtree = KDTree(data_points)
    if method[keys[k]] == "sum":
        max_dist=0.242487*2 # hexagon ray
        print("max_dist", max_dist)
        neighbors_indices = kdtree.query_ball_point(wanted_points, max_dist)
    else: #default is mean
        nb_neighbors=max(len(data_points)//len(wanted_ids),2)
        print("nb_neighbors", nb_neighbors)
        distances, neighbors_indices = kdtree.query(wanted_points, k=nb_neighbors)

    neig=[len(x) for x in neighbors_indices]
    print(np.mean(neig), max(neig), min(neig), np.percentile(neig,85))

    # Get the values of the closest neighbors
    closest_values = [[data[k][data_points[j]] for j in neighbors_indices[i]] for i in range(len(wanted_points))]

    # Calculate the mean/sum of the values
    if method[keys[k]] == "sum":
        result[keys[k]] = np.array([sum(v) for v in closest_values])
    else: # default is mean
        result[keys[k]] = np.mean(closest_values, axis=1)

print(result["animals"].shape)

print(np.mean(result["plants"]))

df = pd.DataFrame(result, index=wanted_ids)
print(df)

# plt.scatter(raw_data["animals"][:,1],raw_data["animals"][:,2],c=raw_data["animals"][:,0],marker = '_')
# plt.scatter(raw_data["plants"][:,2],raw_data["plants"][:,1],marker = '_')

x,y,z=[],[],[]
for i in range(len(result["plants"])):
    key=wanted_points[i]
    val=result["plants"][i]
    x.append(key[0])
    y.append(key[1])
    z.append(val)

fig = plt.figure()
ax = fig.add_subplot()
sct = ax.scatter(x, y, c=z, marker='o', cmap=cm.copper, s=1)
ax.set_title('Figure 1. Geodistribution of crop wild relatives')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.colorbar(sct, orientation="vertical")
plt.show()

df.to_csv("world_data_biodiversity.csv")
