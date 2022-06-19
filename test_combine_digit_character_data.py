import tensorflow as tf #not use
import numpy as np #using to change type, concatenate, reshape, make array, find argmax, arrange, save
import pandas as pd #using to read data from csv file
import cv2 #not use
from keras.utils import np_utils #using to covert the label (target) to categorical formats for CNN
from keras.datasets import mnist #import mnist dataset using keras.datasets
from sklearn.utils import shuffle ##using to shuffle dataset (for show plot)
from sklearn.model_selection import train_test_split #using to split data to train and test
from matplotlib import pyplot as plt ##using to plot data

#  Note for each step
## Note for the step that not necessary, can delete if you want

# Saved aussie handwritten database A_z Handwritten Data
# Store the data to variable data_root
data_root = "D:\Code\Code\Workspace\Python\A_Z Handwritten Data\A_Z Handwritten Data.csv"

# Because the data is a csv file, using pandas library to read AZ characters dataset in the type of float 32
# Store to variable "dataset"
dataset = pd.read_csv(data_root).astype(np.float32)
# Each particular image has separate into 785 columns (from 1 column label at first and 28*28 pixel)
# Rename '0' column of "dataset" as "label"
dataset.rename(columns = {'0': "label"}, inplace = True)

# Dropped labels and captured the rest of 784 pixels of each image in "dataset" and passed it to "letter_x" variable
# Axis = 1, drop with column (row if axis = 0)
letter_x = dataset.drop("label", axis = 1)
# Captured the labels of "dataset" and passed it to "letter_y" variable
letter_y = dataset["label"]
# Load mnist data and set to "digit_train_x, digit_train_y", "digit_test_x, digit_test_y" variables
(digit_train_x, digit_train_y), (digit_test_x, digit_test_y) = mnist.load_data()

# Passed data values of "letter_x" to "letter_x"
letter_x = letter_x.values

## Print out the shape of letter and digit "dataset" 
print("\nShape of \"letter_x\" and \"letter_y\"")
print(letter_x.shape, letter_y.shape)
print("Shape of \"digit_train_x\" and \"digit_train_y\"")
print(digit_train_x.shape, digit_train_y.shape)
print("Shape of \"digit_test_x\" and \"digit_test_y\"")
print(digit_test_x.shape, digit_test_y.shape)

# Remember x is number of data, type of data (784 columns or 28*28 columns and rows)
# Remember y is number of labels
# Concatenate (definition: link (things) together in a chain of a series)
# => Concatenated both digit train and digit test
# Store in "digit_data" and "digit_target" variable
digit_data = np.concatenate((digit_train_x, digit_test_x))
digit_target = np.concatenate((digit_train_y, digit_test_y))

## Print out the shape new digit data set
print("\nAfter concatenate train and test x(data) and y(label)")
print("Shape of \"digit_data\" and \"digit_target\"")
print(digit_data.shape, digit_target.shape)

# Added 26 labels to "digit_target" to merge letter and digit data set (26 labels mean 26 characters)
# From 0 to 26 is for the labels of A to Z characters, from 26 to onwards is for labels of all 0 to 9 digit numbers
digit_target += 26

# Look through each and every feature images of A to Z handwritten data set
# Create a "data" list variable
data = []
# Look through each and every 784 pixels vector of "letter_x" and assign it to a "flatten" variable in each time
for flatten in letter_x:
    # reshape "flatten" with 784 pattern pixels to 28 and 28 and 1 shape (three dimensional array)
    # store in "image" variable
    image = np.reshape(flatten, (28,28,1))
    # using numpy.36 that reshape image is going to be append to "data" list  
    data.append(image)

# Converted "data" list as a numpy array in the type of float 32
# Store in "letter_data" variable
letter_data = np.array(data, dtype = np.float32)
# Assign label of A to Z handwritten data set into "letter_target"
# Store in "letter_target" variable
letter_target = letter_y

## Check the respect related of digit_data size
print(digit_data.shape[0])
print(digit_data.shape[1])
print(digit_data.shape[2])
# Reshape "digit_data" as a 4 dimensional array using numpy.reshape
digit_data = np.reshape(digit_data, (digit_data.shape[0], digit_data.shape[1], digit_data.shape[2], 1))

## Print out the shape new digit and letter data set
print("\nShape of \"letter_data\" and \"letter_target\"")
print(letter_data.shape, letter_target.shape)
print("Shape of \"digit_data\" and \"digit_target\"")
print(digit_data.shape, digit_target.shape)

## shuffle and randomly picked 100 images from A to Z hand written character data in "letter_data"
## Store in shuffled_data variable
shuffled_data = shuffle(letter_data)
## Set row and columns to show data
rows, cols = 10, 10
## Plot data using matplotlib
plt.figure(figsize = (20,20))
for i in range(rows * cols):
    plt.subplot(cols, rows, i+1)
    plt.imshow(shuffled_data[i].reshape(28,28), interpolation = "nearest", cmap = "gray")
plt.show();

## shuffle and randomly picked 100 images from 0 to 9 hand written character data in "digit_data"
## Store in shuffled_data variable
shuffled_data = shuffle(digit_data)
## Set row and columns to show data
rows, cols = 10, 10
## Plot data using matplotlib
plt.figure(figsize = (20,20))
for i in range(rows * cols):
    plt.subplot(cols, rows, i+1)
    plt.imshow(shuffled_data[i].reshape(28,28), interpolation = "nearest", cmap = "gray")
plt.show();

# Concatenated "letter_data" and "digit_data" as "data"
data = np.concatenate((letter_data, digit_data))
# Concatenated "letter_target" and "digit_target" as "target"
target = np.concatenate((letter_target, digit_target))

## Print out the shape of "data" and "target" data set
print("\nAfter concatenate letter_data, digit_data and letter_target, digit_target (target = label)")
print("Shape of \"data\" and \"target\"")
print(data.shape, target.shape)

# shuffle and randomly picked 100 images from hand written character data in "data"
## Store in shuffled_data variable
shuffled_data = shuffle(data)
## Set row and columns to show data
rows, cols = 10, 10
## Plot data using matplotlib
plt.figure(figsize = (20,20))
for i in range(rows * cols):
    plt.subplot(cols, rows, i+1)
    plt.imshow(shuffled_data[i].reshape(28,28), interpolation = "nearest", cmap = "gray")
plt.show();

# Split merge data set into train_data, test_data and train_labels, test_labels
train_data, test_data, = train_test_split(data, test_size = 0.2)
train_labels, test_labels = train_test_split(target, test_size = 0.2) 

## Print out the shape of "data" and "label" data set after split
print("After split merge data set into train_data, test_data and train_labels, test_labels")
print("\nShape of \"train_data\" and \"train_labels\"")
print(train_data.shape, train_labels.shape)
print("Shape of \"test_data\" and \"test_labels\"")
print(test_data.shape, test_labels.shape)

# Normalized each pixel in "train_data" and "test_data" by dividing them by 255 
# Divide by 255 meaning normalized each pixel "train_data" and "test_data" in the range of 0 to 1
train_data /= 255
test_data /= 255

# Converted "train_labels" and "test_labels" to the categorical formation
# Since the convolution neural network (CNN) accepts only the labels of categorical formats
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

## Print out the shape of train and test
print("\nAfter Converted \"train_labels\" and \"test_labels\" to the categorical formation")
print("Shape of \"train_data\" and \"train_labels\"")
print(train_data.shape, train_labels.shape)
print("Shape of \"test_data\" and \"test_labels\"")
print(test_data.shape, test_labels.shape)


# Covert "train_data" and "test_data" into 4 dimensional arrays 
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))

## Print out the shape of train and test
print("\nAfter convert train_data and test_data in to 4D arrays")
print("Shape of \"train_data\" and \"train_labels\"")
print(train_data.shape, train_labels.shape)
print("Shape of \"test_data\" and \"test_labels\"")
print(test_data.shape, test_labels.shape)

## plot the train and test distribution table
## Create counts variable for train and test label
train_labels_counts = [0 for i in range(36)]
test_labels_counts = [0 for i in range(36)]
## Count number of data in each label
for i in range(train_data.shape[0]):
    train_labels_counts[np.argmax(train_labels[i])] += 1
for i in range(test_data.shape[0]):
    test_labels_counts[np.argmax(test_labels[i])] += 1
## plot
frequency = [train_labels_counts, test_labels_counts]

fig =  plt.figure(figsize=(8, 6))
ax = fig.add_axes([0, 0, 1, 1])
x = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.xticks(range(len(frequency[0])), x)
plt.title("train vs. test data distribution")
plt.xlabel("character")
plt.ylabel("frequency")

ax.bar(np.arange(len(frequency[0])), frequency[0], color="b", width=0.35)
ax.bar(np.arange(len(frequency[1])) + 0.35, frequency[1], color="r", width=0.35)
ax.legend(labels=["train", "test"])
plt.show()

# Save train, test data and labels
np.save("D:\Code\Code\Workspace\Python\A_Z Handwritten Data\\numpy\\train_data", train_data)
np.save("D:\Code\Code\Workspace\Python\A_Z Handwritten Data\\numpy\\train_labels", train_labels)
np.save("D:\Code\Code\Workspace\Python\A_Z Handwritten Data\\numpy\\test_data", test_data)
np.save("D:\Code\Code\Workspace\Python\A_Z Handwritten Data\\numpy\\test_labels", test_labels)