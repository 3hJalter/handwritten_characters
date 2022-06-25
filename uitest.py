
import tkinter as tk
from turtle import width
from PIL import Image, ImageDraw
import cv2 as cv
from matplotlib.pyplot import text
import numpy as np
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

class DrawCharacter:
    def __init__(self, parent, x_pos, y_pos):
        self.parent = parent
        self.x_pos = x_pos
        self.y_pos = y_pos
        
        self.b1 = "up"
        self.xold = None
        self.yold = None

        self.drawing_area = tk.Canvas(
            self.parent, bg='black', width=200, height=200)
        self.drawing_area.place(x=self.x_pos, y=self.y_pos + 20)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)

        self.result_area = tk.Canvas(self.parent, width= 200, height= 200, bg = 'black')
        self.result_area.place(x=self.x_pos + 220, y=self.y_pos + 20)

        self.save_btn = tk.Button(self.parent, text="Save", width=10, bg='white', command=self.save)
        self.save_btn.place(x=200/7, y=240)

        self.clear_btn = tk.Button(self.parent, text="Clear", width=10, bg='white', command=self.clear)
        self.clear_btn.place(x=(200/7)+90, y=240)

        self.proceed_btn = tk.Button(self.parent, text="Proceed", width=10, bg='white', command=self.proceed)
        self.proceed_btn.place(x=295, y=240)

        self.image = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def proceed(self):
        img = cv.imread(f'tmp.png', cv.IMREAD_GRAYSCALE)
        img = np.invert(np.array(img))
        img = cv.resize(img, (28, 28))
        self.result_area.create_text(100, 100, text = predict(model, img), fill = 'white', font=('Helvetica','12','bold'))

    def reset(self):
        self.result_area.delete("all")

    def save(self):
        self.reset()
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
                    ((self.xold, self.yold), (event.x, event.y)), (0, 128, 0), width=20)

        self.xold = event.x
        self.yold = event.y

def load_model(path):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (5, 5), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(36, activation="softmax"))

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

model = load_model("D:\Project\Handwriting\\best_val_loss_model.h5")      

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Handwritten Recognition')

    lb1 = tk.Label(root, text = 'Drawing Area', bg = 'white')
    lb1.place(x = 74, y = 4)

    lb2 = tk.Label(root, text = 'Result', bg = 'white')
    lb2.place(x = 315, y = 4)

    root.wm_geometry("%dx%d+%d+%d" % (445, 290, 10, 10))
    root.config(bg='grey')
    DrawCharacter(root, 10, 10)

    root.mainloop()

    # img = cv.imread(f'tmp.png', cv.IMREAD_GRAYSCALE)
    # img = np.invert(np.array(img))
    # img = cv.resize(img, (28, 28))

    # print(predict(model, img))

    
   
   
