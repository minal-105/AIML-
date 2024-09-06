import tkinter as tk
from tkinter import Canvas, Button, Label, IntVar
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.datasets import mnist

print("Loading and preparing data...")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

print("Defining the model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training the model... This may take a few minutes.")
model.fit(train_images, train_labels, epochs=5, validation_split=0.1, verbose=1)

class WhiteboardDigitRecognizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Whiteboard Digit Recognizer")
        
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(pady=10)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_xy)
        
        self.predict_button = Button(self.master, text="Predict", command=self.predict_digit)
        self.predict_button.pack(pady=5)
        
        self.clear_button = Button(self.master, text="Clear Whiteboard", command=self.clear_canvas)
        self.clear_button.pack(pady=5)
        
        self.invert_var = IntVar()
        self.invert_check = tk.Checkbutton(self.master, text="Invert Colors", variable=self.invert_var)
        self.invert_check.pack(pady=5)
        
        self.result_label = Label(self.master, text="Write a digit on the whiteboard")
        self.result_label.pack(pady=5)
        
        self.accuracy_labels = []
        for i in range(3):
            label = Label(self.master, text="")
            label.pack(pady=2)
            self.accuracy_labels.append(label)
        
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color="black")
        self.draw = ImageDraw.Draw(self.image)
        
        self.old_x = None
        self.old_y = None

    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                    width=20, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                           fill="white", width=20)
        self.old_x = event.x
        self.old_y = event.y
        self.predict_digit()  # Predict in real-time as user draws

    def reset_xy(self, event):
        self.old_x = None
        self.old_y = None

    def preprocess_image(self):
        img = self.image.copy()
        if self.invert_var.get():
            img = ImageOps.invert(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.filter(ImageFilter.SHARPEN)
        img = ImageOps.autocontrast(img)
        
        # Find bounding box
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        
        # Resize and pad to 28x28
        img = ImageOps.fit(img, (20, 20), Image.LANCZOS)
        img_padded = Image.new('L', (28, 28), 0)
        img_padded.paste(img, (4, 4))
        
        return np.array(img_padded).reshape(1, 28, 28, 1) / 255.0

    def predict_digit(self):
        img_array = self.preprocess_image()
        prediction = model.predict(img_array)[0]
        
        top_3 = np.argsort(prediction)[-3:][::-1]
        
        self.result_label.config(text=f"Top Prediction: {top_3[0]} (Confidence: {prediction[top_3[0]]:.2%})")
        
        for i, digit in enumerate(top_3):
            confidence = prediction[digit]
            self.accuracy_labels[i].config(text=f"{i+1}. Digit {digit}: {confidence:.2%}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Write a digit on the whiteboard")
        for label in self.accuracy_labels:
            label.config(text="")

if __name__ == "__main__":
    print("Initializing GUI...")
    root = tk.Tk()
    app = WhiteboardDigitRecognizer(root)
    root.mainloop()