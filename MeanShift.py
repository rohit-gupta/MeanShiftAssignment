from time import time
import math

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# from sklearn.cluster import KMeans
# from sklearn.metrics import pairwise_distances_argmin
# from sklearn.datasets import load_sample_image
# from sklearn.utils import shuffle

from skimage.color import rgb2gray
from skimage import data

from scipy import spatial

def euclidean_dist(X1, X2):
    #if(len(X1) != len(X2)):
    #    raise Exception("Euclidean distance is only defined for coordinates of equal dimensionality !")
    X1 = [X1]
    X2 = [X2]
    total = float(0)
    for dimension in range(0, len(X1)):
        total += (X1[dimension] - X2[dimension])**2
    return math.sqrt(total)



def gaussian_kernel(distance, bandwidth = 20):
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
    return val



def mean_shift(window_values, window_position):
    kernel_values = [gaussian_kernel(euclidean_dist(pixel,window_position)) for pixel in window_values]
    return np.dot(kernel_values,window_values)/np.sum(kernel_values)


def predict(clusters, point):
    distances = [gaussian_kernel(euclidean_dist(point,float(mode)/255)) for mode in clusters]
    return clusters[np.argmax(distances)]



NUM_WINDOWS = 64
WINDOW_SIZE = float(16)
NUM_ITERATIONS = 50

initial_window_positions = np.arange(WINDOW_SIZE/2,255-(WINDOW_SIZE/2),(256/NUM_WINDOWS))

window_positions= initial_window_positions


img = data.lena()
img_gray = rgb2gray(img)
gray_image = np.array(img_gray, dtype=np.float64)

sorted_img = np.sort(gray_image.flatten())

t0 = time()

for _ in xrange(1,NUM_ITERATIONS):
    new_position = []
    print "Iteration:\t" + str(_)
    for index,position in enumerate(window_positions):
        left = np.searchsorted(sorted_img*255, (position - (WINDOW_SIZE/2)), 'left')
        right = np.searchsorted(sorted_img*255,(position + (WINDOW_SIZE/2)), 'right')
        new_position.append(mean_shift(sorted_img[left:right],position/255)*255) 
    window_positions = new_position



tN = time()

print "Time:\t" + str(tN - t0)



plt.hist(gray_image.flatten(), 256, normed=1)
plt.plot([_/255 for _ in window_positions],[0.5]*len(window_positions), marker='o', color='r', ls='')
plt.plot([_/255 for _ in initial_window_positions],[0.5]*len(window_positions), marker='*', color='g', ls='')

plt.show()

clusters = np.unique([int(_) for _ in window_positions[1:]])

print clusters

newimage=np.empty(np.shape(gray_image.flatten()))
for index,pixel in enumerate(gray_image.flatten()):
    newimage[index]=predict(clusters, pixel) 


newimage=np.reshape(newimage,gray_image.shape)

#print gray_image.shape

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Mean Shift Segmentation')
plt.imshow(newimage, cmap = cm.Greys_r)

plt.show()

# n_colors = 64

# # Load the Summer Palace photo
# china = load_sample_image("china.jpg")

# # Convert to floats instead of the default 8 bits integer coding. Dividing by
# # 255 is important so that plt.imshow behaves works well on float data (need to
# # be in the range [0-1]
# china = np.array(china, dtype=np.float64) / 255

# # Load Image and transform to a 2D numpy array.
# w, h, d = original_shape = tuple(china.shape)
# assert d == 3
# image_array = np.reshape(china, (w * h, d))

# print("Fitting model on a small sub-sample of the data")
# t0 = time()
# image_array_sample = shuffle(image_array, random_state=0)[:1000]
# kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
# print("done in %0.3fs." % (time() - t0))

# # Get labels for all points
# print("Predicting color indices on the full image (k-means)")
# t0 = time()
# labels = kmeans.predict(image_array)
# print("done in %0.3fs." % (time() - t0))


# codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
# print("Predicting color indices on the full image (random)")
# t0 = time()
# labels_random = pairwise_distances_argmin(codebook_random,
#                                           image_array,
#                                           axis=0)
# print("done in %0.3fs." % (time() - t0))


# def recreate_image(codebook, labels, w, h):
#     """Recreate the (compressed) image from the code book & labels"""
#     d = codebook.shape[1]
#     image = np.zeros((w, h, d))
#     label_idx = 0
#     for i in range(w):
#         for j in range(h):
#             image[i][j] = codebook[labels[label_idx]]
#             label_idx += 1
#     return image

# # Display all results, alongside original image
# plt.figure(1)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Original image (96,615 colors)')
# plt.imshow(china)

# plt.figure(2)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Quantized image (64 colors, K-Means)')
# plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

# plt.figure(3)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Quantized image (64 colors, Random)')
# plt.imshow(recreate_image(codebook_random, labels_random, w, h))
# plt.show()

