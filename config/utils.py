# Some references
# https://github.com/rexledesma/Single-Image-Haze-Removal-Using-Dark-Channel-Prior/blob/master/haze-removal.ipynb
import cv2
import numpy as np
from scipy import ndimage

def dark_channel(img, window):
    r, g, b = cv2.split(img)
    dark = ndimage.minimum_filter(img, footprint=np.ones((window, window, 3)), mode='nearest')
    return dark[:,:,1]

def atmospheric_light(img, window):
	size = img.shape[:2]
	k = int(0.001 * np.prod(size))
	dark = dark_channel(img=img, window=window)
	index = np.argpartition(-dark.ravel(), k)[:k]
	x, y = np.hsplit(np.column_stack(np.unravel_index(index, size)), 2)
	A = np.array([img[x, y, 0].max(), img[x, y, 1].max(), img[x, y, 2].max()])
	return A

def transmission(img, omega=0.95, window=20):
	A = atmospheric_light(img, window)
	norm = img / A
	dark = dark_channel(norm, window)
	trans = 1 - omega * dark
	return trans
