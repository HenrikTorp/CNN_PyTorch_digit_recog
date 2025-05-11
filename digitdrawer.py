"""
    This module contains the DigitDrawer class, which is responsible for drawing
    digits on a canvas and recognizing them using a neural network.
    The images are resized to 28x28 pixels to match the MNIST dataset format.
"""

import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
from models.cnn import CNNModel  # Import your PyTorch CNN model class
import os
from datetime import datetime


class DigitDrawer:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Drawer")
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white")

        # Add a faint rectangle in the center of the canvas
        margin = 50  # Size of the rectangle
        self.canvas.create_rectangle(
            margin, margin, self.canvas_width - margin, self.canvas_height - margin,
            outline="lightgray", dash=(2, 2)
        )

        # Pack the canvas
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)  # White background
        self.draw = ImageDraw.Draw(self.image)
        self.button_frame = tk.Frame(master)
        self.button_frame.pack()
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)
        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT)

        # Load the trained neural network model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = CNNModel().to(self.device)  # Initialize your CNN model
        self.nn.load_state_dict(torch.load("models/cnn_model.pth", map_location=self.device))
        self.nn.eval()  # Set the model to evaluation mode

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Add a faint rectangle in the center of the canvas
        margin = 50  # Size of the rectangle
        self.canvas.create_rectangle(
            margin, margin, self.canvas_width - margin, self.canvas_height - margin,
            outline="lightgray", dash=(2, 2)
        )

    def predict_digit(self):
        # Preprocess the image to match the MNIST dataset format
        original_image = self.image.copy()
        resized_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        input_array = np.array(resized_image, dtype=np.float32)

        # Invert the colors (white background to black and vice versa)
        input_array = 255 - input_array
        input_array = input_array / 255.0  # Normalize to [0, 1]
        # renormalize to [-1, 1]
        input_array = (input_array - 0.5) / 0.5

        # Convert to PyTorch tensor and add batch dimension
        input_tensor = torch.tensor(input_array).unsqueeze(0).unsqueeze(0).to(self.device)

        # Make a prediction using the neural network
        with torch.no_grad():
            output = self.nn(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()

        # Clear any previous label from the canvas
        self.canvas.delete("label")

        # Display the predicted label at the top of the canvas
        self.canvas.create_text(
            self.canvas_width // 2, 10,  # Position: center horizontally, 10 pixels from the top
            text=f"Predicted: {predicted_label}",
            fill="blue",
            font=("Helvetica", 16, "bold"),
            tags="label"  # Add a tag to easily delete/replace this text later
        )

        # Save the preprocessed image (optional)
        save_image = False  # Set to True if you want to save the image
        if save_image:
            save_folder = "data/images"
            os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Unique timestamp
            save_path = os.path.join(save_folder, f"digit_{timestamp}.png")
            original_image.save(save_path)  # Save the original image
            print(f"Saved image to {save_path}")

        print(f"Predicted Label: {predicted_label}")


if __name__ == "__main__":
    root = tk.Tk()
    digit_drawer = DigitDrawer(root)
    root.mainloop()



