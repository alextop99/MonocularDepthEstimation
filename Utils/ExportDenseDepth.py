import tensorflow as tf
import cv2
import skimage
import numpy as np
from scipy.sparse.linalg import spsolve
import scipy
from dataset.kitti_dataset import KittiDatasetGenerate
from matplotlib import pyplot as plt
import threading
import os

THREADS = 12
SIZE = None

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
	imgIsNoise = imgDepthInput == 0
	maxImgAbsDepth = np.max(imgDepthInput)
	imgDepth = imgDepthInput / maxImgAbsDepth
	imgDepth[imgDepth > 1] = 1
	(H, W) = imgDepth.shape
	numPix = H * W
	indsM = np.arange(numPix).reshape((W, H)).transpose()
	knownValMask = (imgIsNoise == False).astype(int)
	grayImg = skimage.color.rgb2gray(imgRgb)
	winRad = 1
	len_ = 0
	absImgNdx = 0
	len_window = (2 * winRad + 1) ** 2
	len_zeros = numPix * len_window

	cols = np.zeros(len_zeros) - 1
	rows = np.zeros(len_zeros) - 1
	vals = np.zeros(len_zeros) - 1
	gvals = np.zeros(len_window) - 1

	for j in range(W):
		for i in range(H):
			nWin = 0
			for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
				for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
					if ii == i and jj == j:
						continue

					rows[len_] = absImgNdx
					cols[len_] = indsM[ii, jj]
					gvals[nWin] = grayImg[ii, jj]

					len_ = len_ + 1
					nWin = nWin + 1

			curVal = grayImg[i, j]
			gvals[nWin] = curVal
			c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

			csig = c_var * 0.6
			mgv = np.min((gvals[:nWin] - curVal) ** 2)
			if csig < -mgv / np.log(0.01):
				csig = -mgv / np.log(0.01)

			if csig < 2e-06:
				csig = 2e-06

			gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
			gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
			vals[len_ - nWin:len_] = -gvals[:nWin]

			rows[len_] = absImgNdx
			cols[len_] = absImgNdx
			vals[len_] = 1

			len_ = len_ + 1
			absImgNdx = absImgNdx + 1

	vals = vals[:len_]
	cols = cols[:len_]
	rows = rows[:len_]
	A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	rows = np.arange(0, numPix)
	cols = np.arange(0, numPix)
	vals = (knownValMask * alpha).transpose().reshape(numPix)
	G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	A = A + G
	b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

	new_vals = spsolve(A, b)
	new_vals = np.reshape(new_vals, (H, W), 'F')

	denoisedDepthImg = new_vals * maxImgAbsDepth
    
	output = denoisedDepthImg.reshape((H, W)).astype('float32')

	output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
	return output

#* Load images from paths
def load(image_path, depth_path):
    image_ = cv2.imread(image_path)
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    image_ = tf.image.convert_image_dtype(image_, tf.float32)

    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if not os.path.isfile(depth_path.replace("\\","/").replace(".png", "_depth.png")):
        output = fill_depth_colorization(image_, depth_map)
		
        plt.imsave(depth_path.replace("\\","/").replace(".png", "_depth.png"), output, cmap='gray')

kittiDataset = KittiDatasetGenerate("dataset/kitti/")
trainData, valData = kittiDataset.load_train_val()

if not SIZE is None:
	trainData = trainData[:SIZE]

sizePerThread = int(len(trainData) / THREADS)

def ThreadFunc(start, end, data):
    for index, row in data[start:end].iterrows():
        print(index, end)
        load(row['image'], row['depth_map'])

t = []
for i in range(THREADS):
    if i != THREADS - 1:
        t += [threading.Thread(target=ThreadFunc, args=(sizePerThread * i, sizePerThread * (i+1), trainData))]
    else:
        t += [threading.Thread(target=ThreadFunc, args=(sizePerThread * i, len(trainData), trainData))]
  
    t[i].start()

for thread in t:
    thread.join()
    
if not SIZE is None:
	valData = valData[:SIZE]

sizePerThread = int(len(valData) / THREADS)
        
t = []
for i in range(THREADS):
    if i != THREADS - 1:
        t += [threading.Thread(target=ThreadFunc, args=(sizePerThread * i, sizePerThread * (i+1), valData))]
    else:
        t += [threading.Thread(target=ThreadFunc, args=(sizePerThread * i, len(valData), valData))]
  
    t[i].start()

for thread in t:
    thread.join()