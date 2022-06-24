from operator import mod
from statistics import mode
import tkinter as tk
from PIL import Image, ImageDraw
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

class ImageGenerator:
    def __init__(self, parent, posx, posy):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 200
        self.sizey = 200
        self.b1 = "up"
        self.xold = None
        self.yold = None

        self.drawing_area = tk.Canvas(
            self.parent, bg='black', width=self.sizex, height=self.sizey)
        self.drawing_area.place(x=self.posx, y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)

        self.button = tk.Button(self.parent, text="Done!",
                                width=10, bg='white', command=self.save)
        self.button.place(x=self.sizex/7, y=self.sizey+20)

        self.button1 = tk.Button(
            self.parent, text="Clear!", width=10, bg='white', command=self.clear)
        self.button1.place(x=(self.sizex/7)+90, y=self.sizey+20)

        self.image = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def save(self):
        filename = "tmp.png"
        self.image.save(filename)

    def clear(self):
        self.drawing_area.delete("all")
        self.image = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def b1down(self, event):
        self.b1 = "down"

    def b1up(self, event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self, event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(
                    self.xold, self.yold, event.x, event.y, smooth='true', width=10, fill='white')
                self.draw.line(
                    ((self.xold, self.yold), (event.x, event.y)), (0, 128, 0), width=10)

        self.xold = event.x
        self.yold = event.y

def load_model(path):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
            activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
            activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())

    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))

    model.add(Dense(26, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.load_weights(path)
    
    return model

def predict(model, image):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    image = image / 255.0
    image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
    prediction = model.predict(image)
    best_predictions = dict()
    
    for i in range(3):
        max_i = np.argmax(prediction[0])
        acc = round(prediction[0][max_i], 1)
        if acc > 0:
            label = labels[max_i]
            best_predictions[label] = acc
            prediction[0][max_i] = 0
        else:
            break
            
    return best_predictions

model = load_model("D:\Project\Handwriting\handwritten-character-recognition-code\\best-trained-model(.h5-file)\model_hand.h5")      

if __name__ == "__main__":
    root = tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (220, 290, 10, 10))
    root.config(bg='grey')
    ImageGenerator(root, 10, 10)

    root.mainloop()

    img = cv.imread(f'tmp.png', cv.IMREAD_GRAYSCALE)
    img = np.invert(np.array(img))
    img = cv.resize(img, (28, 28))

    plt.imshow(predict(model, img))
   
   
