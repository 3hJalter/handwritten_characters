from tabnanny import check ## Not use
import numpy as np # Use to load the pre-process data ## find argmax and unique
import visualkeras as vk ## Use to visualize the neuron network layer
import pandas as pd ## using to convert data from confusion matrix to plot its
import seaborn as sn ## using to plot confusion matrix
from matplotlib import pyplot as plt ## using to plot loss, val_loss frequency and confusion matrix
from sklearn.metrics import confusion_matrix ## Using to create confusion matrix
from keras.callbacks import ModelCheckpoint # Using to create checkpoint for early stopping and store best_loss_validation value
from keras.models import Sequential # Using to make model into CNN (model of CNN always is sequential)
from keras.callbacks import ReduceLROnPlateau, EarlyStopping # Using them if not have enough memory to call method in line 111
from keras.preprocessing.image import ImageDataGenerator ## Not use
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # Using to make layer

#  Note for each step
## Note for the step that unnecessary, can delete if you want

# Import train_data, train_labels, test_data and test_labels
train_data = np.load("D:\Code\Code\Workspace\Python\A_Z Handwritten Data\\numpy\\train_data.npy")
train_labels = np.load("D:\Code\Code\Workspace\Python\A_Z Handwritten Data\\numpy\\train_labels.npy")
test_data = np.load("D:\Code\Code\Workspace\Python\A_Z Handwritten Data\\numpy\\test_data.npy")
test_labels = np.load("D:\Code\Code\Workspace\Python\A_Z Handwritten Data\\numpy\\test_labels.npy")

# Because the model of CNN is sequential
model = Sequential()

# a CNN has two sections, first is feature extraction, second is image classification
# First, second and third layer in this code are eligible for feature extraction
# Fourth and fifth layer are eligible for image classification

# => First layer will be a 2D convolution (Conv2D) layer of 32 convolutions filters and kernel_size 5*5
# => Input shape is 28*28 and 1 (28*28 resolution grayscale image)
# => Value activation function is relu function
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
# Added Batch Normalization for the layer to improve the speed up training
# See more why use Batch Normalization at: https://youtu.be/DtEq44FTPM4
model.add(BatchNormalization())
# Second Layer
model.add(Conv2D(32, (5, 5), activation="relu"))
model.add(BatchNormalization())
# Add Max Pooling 2D layer of size 2*2
model.add(MaxPooling2D(2, 2))
# Add Dropout layer of 0.25 dropper probability to avoid from over fitting problems
model.add(Dropout(0.25))
# Third Layer
model.add(BatchNormalization())
# Add flattened layer to flatten every output from batch normalization layer
model.add(Flatten())

# Fourth layer
# Add dense layer of 256 neurons with activation function is relu function
model.add(Dense(256, activation="relu"))
# Fifth layer
# Add output dense layer of 35 neurons (mean 36 labels from CNN or 36 characters and digits) 
# with activation function is soft max function
model.add(Dense(36, activation="softmax"))

# Compile model with the loss function of categorical cross entropy 
# Categorical cross entropy is most commonly used for classification problems
# Pass "adam" optimizer as this model optimizer
# Pass "accuracy" to monitor accuracy whenever the model is going to be framed  
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
## Print the summary of the model
model.summary()

## Visualize the neuron network layer
vk.layered_view(model)

#### Use this code if have enough memory ####
# Added two checkpoints
# Why checkpoint: whenever the model is be trained, or have a minimum loss model, or minimum validation loss model 
# => Should save the model to local machine and it will be the best model 
# Read the link, find keras callbacks: https://miai.vn/2020/09/05/keras-callbacks-tro-thu-dac-luc-khi-train-models/

# First checkpoint: representing minimum loss 
best_loss_checkpoint = ModelCheckpoint(
    filepath= "D:\Code\Code\Workspace\Python\A_Z Handwritten Data\models\\best_loss_model.h5",
    monitor = "loss",
    save_best_only= True,
    save_weights_only= True,
    mode= "min"    
)
# Second checkpoint: representing minimum validation loss 
best_val_loss_checkpoint = ModelCheckpoint(
    filepath = "D:\Code\Code\Workspace\Python\A_Z Handwritten Data\models\\best_val_loss_model.h5",
    monitor = "val_loss",
    save_best_only= True,
    save_weights_only= True,
    mode= "min"
)

# Train model with .fit function
history = model.fit(
    # Pass train_data and train_labels (the pre-process data saved before)
    train_data,
    train_labels,
    # Pass the pre-process test data and labels to validation_data
    validation_data=(test_data, test_labels),
    # Train model with 10 epochs and have batch size is 200
    # Definition of epochs and batch size: https://www.phamduytung.com/blog/2018-10-02-understanding-epoch-batchsize-iterations/
    epochs=10,
    batch_size=50,
    # Pass callbacks to Early stopping by two checkpoints
    # Read Early stopping method for more detail
    # The two checkpoints will capture the minium loss or minimum validation loss of train model
    # then find the best model min_loss and min_val_loss model and saved to local machine
    callbacks=[best_loss_checkpoint, best_val_loss_checkpoint]
)
#### If not have enough memory, do the code below ####


# # Search with key word "Reduce learning rate" and "Early stopping" for more detail
# # Reduce learning rate when a metric has stopped improving.
# # Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 
# # This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
# # Stop training when a monitored metric has stopped improving.
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# print("Check if exceed memory relate to best_loss and best_val_loss or relate to model.fit\n")
# # Train model with .fit function
# history = model.fit(
#     # Pass train_data and train_labels (the pre-process data saved before)
#     train_data,
#     train_labels,
#     # Pass the pre-process test data and labels to validation_data
#     validation_data=(test_data, test_labels),
#     # Train model with 10 epochs and have batch size is 200
#     # Definition of epochs and batch size: https://www.phamduytung.com/blog/2018-10-02-understanding-epoch-batchsize-iterations/
#     epochs=10,
#     batch_size=50,
#     # Pass callbacks to Early stopping by two checkpoints
#     # Read Early stopping method for more detail
#     callbacks=[reduce_lr, early_stop]
# )

## See the result
## Plot loss and val_loss frequency in each epoch 
plt.plot(history.history["loss"], 'b', label="loss")
plt.plot(history.history["val_loss"], 'r', label="val_loss")
plt.xlabel("epoch")
plt.ylabel("frequency")
plt.legend()
plt.show()

## Load the weight of the best validation loss model
model.load_weights("D:\Code\Code\Workspace\Python\A_Z Handwritten Data\models\\best_val_loss_model.h5")

## Evaluate data by passing test_data and test_labels
loss, acc = model.evaluate(test_data, test_labels)
## Print loss and accuracy
print(loss, acc)

## Predict model by passing test_data
predictions = model.predict(test_data)

## Pass test_labels and predictions to confusion matrix
confusion = confusion_matrix(
    np.argmax(test_labels, axis=1),
    np.argmax(predictions, axis=1)
)
## Print the matrix
print(confusion)

## Plot confusion matrix with matplotlib
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

df_cm = pd.DataFrame(confusion, columns=np.unique(labels), index = np.unique(labels))
df_cm.index.name = 'actual'
df_cm.columns.name = 'predicted'
plt.figure(figsize = (20,15))
sn.set(font_scale=1.4) 
sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 15}, fmt="d")
