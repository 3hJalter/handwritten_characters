import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

path = r'D:\Project\Handwriting\Digit\1.png'

# Using cv2.imread() method
img = cv.imread(path)

# Displaying the image
cv.imshow('image', img)
