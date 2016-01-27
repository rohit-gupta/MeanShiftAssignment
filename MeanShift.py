from time import time
import math
from sys import exit

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
    if(len(X1) != len(X2)):
       raise Exception("Euclidean distance is only defined for coordinates of equal dimensionality !")
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


def mean_shift_vector(window_values, window_position):
    kernel_values = [gaussian_kernel(euclidean_dist(pixel,window_position)) for pixel in window_values]
    kernel_values = np.array(kernel_values).reshape(1,len(kernel_values))
    return np.dot(kernel_values,window_values)/np.sum(kernel_values) # TODO


def predict(clusters, point):
    distances = [gaussian_kernel(euclidean_dist(point,float(mode)/256)) for mode in clusters]
    return clusters[np.argmax(distances)]


def normalize_position(position, width, height):
    normalized_position = [None]*3
    normalized_position[0] = position[0]/256
    normalized_position[1] = position[1]/width
    normalized_position[2] = position[2]/height
    return normalized_position


def denormalize_position(normalized_position, width, height):
    denormalized_position = [None]*3
    denormalized_position[0] = 256*normalized_position.T[0]
    denormalized_position[1] = width*normalized_position.T[1]
    denormalized_position[2] = height*normalized_position.T[2]
    return np.array(denormalized_position).flatten()


# Loading image data
img = data.lena()
img_gray = rgb2gray(img)
gray_image = np.array(img_gray, dtype=np.float64)

# sorted_img = np.sort(gray_image.flatten())

# Creating Image data KDTree
XX,YY = np.meshgrid(np.arange(gray_image.shape[1]),np.arange(gray_image.shape[0]))
img_table = np.vstack((gray_image.ravel(),XX.ravel(),YY.ravel())).T

# Normalization of dimensions
width = np.max(img_table[:,1])
height = np.max(img_table[:,2])

img_table[:,1] = img_table[:,1]/width
img_table[:,2] = img_table[:,2]/height

tree = spatial.KDTree(img_table)

# TODO
# pts = np.array([[0.4, 0.4], [0.5, 0.5]])
# tree.data[tree.query_ball_point([0.1, 100, 100], 5, p=2.0)]

# Hyperparameters
#NUM_WINDOWS = 64
WINDOW_SIZE = float(0.2)
NUM_ITERATIONS = 5

# Initialization
# initial1_window_positions = np.arange(WINDOW_SIZE/2,256-(WINDOW_SIZE/2),(256/NUM_WINDOWS))

intensity_step = 8
min_intensity = np.min(img_table[:,0])*256 + intensity_step/2
max_intensity = np.max(img_table[:,0])*256 - intensity_step/2

width_step = 32
min_width = width_step/2
max_width = width - width_step/2

height_step = 32
min_height = height_step/2
max_height = height - height_step/2

initial_window_positions = np.mgrid[min_intensity:max_intensity:intensity_step, min_width:max_width:width_step, min_height:max_height:height_step].reshape(3,-1).T
window_positions= initial_window_positions



# Mean shift iterations
t0 = time()

for _ in xrange(0,NUM_ITERATIONS):
    new_positions = []
    print "Iteration:\t" + str(_)
    print "Windows:\t" + str(len(window_positions))
    for position in window_positions:
        # left = np.searchsorted(sorted_img*256, (position - (WINDOW_SIZE/2)), 'left')
        # right = np.searchsorted(sorted_img*256,(position + (WINDOW_SIZE/2)), 'right')
        #print position
        normalized_position = normalize_position(position, width, height)
        try:
            neighbours = tree.data[tree.query_ball_point(normalized_position, WINDOW_SIZE, p=2.0)]
        except Exception, e:
            print normalized_position
            raise e
        if len(neighbours):
            shifted_pos = denormalize_position(mean_shift_vector(neighbours,normalized_position), width, height)
            new_positions.append(shifted_pos)
    window_positions = new_positions

tN = time()

print "Time:\t" + str(tN - t0)


# Plotting Results
# plt.hist(gray_image.flatten(), 256, normed=1)
# plt.plot([_/256 for _ in window_positions],[0.5]*len(window_positions), marker='o', color='r', ls='')
# plt.plot([_/256 for _ in initial_window_positions],[0.5]*len(window_positions), marker='*', color='g', ls='')

# plt.show()


# TODO

# Clustering
# clusters = np.unique([int(_) for _ in window_positions[1:]])

# print clusters

# newimage=np.empty(np.shape(gray_image.flatten()))
# for index,pixel in enumerate(gray_image.flatten()):
#     newimage[index]=predict(clusters, pixel) 


# newimage=np.reshape(newimage,gray_image.shape)

# TODO


#print gray_image.shape

# TODO

# plt.figure(1)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Mean Shift Segmentation')
# plt.imshow(newimage, cmap = cm.Greys_r)

# plt.show()

# TODO

# n_colors = 64

# # Load the Summer Palace photo
# china = load_sample_image("china.jpg")

# # Convert to floats instead of the default 8 bits integer coding. Dividing by
# # 256 is important so that plt.imshow behaves works well on float data (need to
# # be in the range [0-1]
# china = np.array(china, dtype=np.float64) / 256

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

