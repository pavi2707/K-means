



import numpy as np
import random
import matplotlib.pyplot as plt

i_data=np.loadtxt('data.txt',dtype= 'int',delimiter=",")
print(i_data)
input_data=np.delete(i_data, np.s_[0], axis=1) 
input_data=np.delete(i_data, np.s_[10], axis=1)
print(input_data)
k_values = range(2,8)
potential_func_vals = np.zeros(6)

for p,k in enumerate(k_values):
    
    centers = random.sample(range(699), k)
    centroids = input_data[centers]
    cluster_label = np.zeros(len(input_data))
    prev_centroids = np.zeros(centroids.shape)
    
    while(np.linalg.norm(centroids - prev_centroids, axis = None) != 0):
        for i,x in enumerate(input_data):
            distances = np.linalg.norm(x - centroids, axis = 1)
            cluster_label[i] = np.argmin(distances)
        
        prev_centroids = np.copy(centroids)
        
        for i in range(k):
            data = [input_data[z] for z in range(len(input_data)) if cluster_label[z] == i]
            centroids[i] = np.mean(data, axis = 0)
            
    Loss = 0
    for i in range(k):
        data = [input_data[z] for z in range(len(input_data)) if cluster_label[z] == i]
        for z in data:
            #Loss = Loss + np.square(np.linalg.norm(centroids[i]-z, axis = None))
            Loss = Loss + np.square(np.linalg.norm(z - centroids[i], axis = None))
            
    potential_func_vals[p] = Loss



print(potential_func_vals)
plt.figure(1)
plt.plot(k_values, potential_func_vals)
plt.xlabel("K value")
plt.ylabel("potential function")
plt.title("K means")
plt.show()

