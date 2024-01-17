from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

import numpy as np

# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()

reference_images = [[],[],[],[],[],[],[],[],[],[]]
mean_reference_images = []

# sort images by label
for idx, i in enumerate(trainy):
	reference_images[i].append(trainX[idx])

for i in range(10):
 # define subplot
 # plt.subplot(330 + 1 + i)
 # plot raw pixel data
 #plt.imshow(np.average(refernces_images[i], axis=0), cmap=plt.get_cmap('gray'))
 mean_reference_images.append(np.average(reference_images[i], axis=0))

for i in range(10):
 # define subplot
 plt.subplot(4,3,i+1)
 # plot raw pixel data
 plt.imshow(mean_reference_images[i], cmap=plt.get_cmap('gray'))

predicted = []

difflist = []
for i in mean_reference_images:
	difflist.append(np.sum( np.abs(trainX[4] - i) ))

print(np.argmin(difflist), trainy[4])


for x in testX:
	difflist = []
	for i in mean_reference_images:
		difflist.append(np.sum( np.abs(x - i) ))

	predicted.append(np.argmin(difflist))

assert len(predicted) == len(testy)
print(accuracy_score(predicted, testy))
plt.show()