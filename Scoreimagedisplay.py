from tkinter import *
from PIL import Image, ImageTk
import os

class MyWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Display Example")
        
        self.img_label = Label(self.root)
        self.img_label.pack()

        self.button = Button(self.root, text="Show Image", command=self.show_image)
        self.button.pack()

    def show_image(self):
        image_path = 'C:/Users/tnlab/OneDrive/Documents/GitHub/AlphaFold/Neurofeedback-Based-BCI/2-Patient Data/N/Pre Evaluation/Score_0.png'  # Replace this with your actual image path

        if not os.path.exists(image_path):
            print("Image file does not exist!")
            return
        pil_image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(pil_image)

        self.img_label.config(image=self.photo)

root = Tk()
window = MyWindow(root)
root.mainloop()