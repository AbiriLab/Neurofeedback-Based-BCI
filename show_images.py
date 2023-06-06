### non-class based, depreciated ### (keeping for now just in case)

import os
from os import listdir
from tkinter import *
from PIL import ImageTk, Image
import time

root = Tk()
root.geometry("800x600")
root.title('Image Slideshow')

image_arr = []
COUNT = 0

# PARAMETERS NEEDED TO BE CHANGED TO CHANGE WHAT IMAGES ARE DISPLAYED
# get the path or directory,  could update this folder dir automatically to get correct images on certain phase
# folder_dir = "C:/Users/tnlab/Neurofeedback-Based-BCI/Images/IndoorScene"
folder_dir= "C:/Users/tnlab/OneDrive/Documents/GitHub/Neurofeedback-Based-BCI/Images/IndoorScene"
TIME_BETWEEN = 10
CURRENT_FOLDER = "IndoorScene"

for images in os.listdir(folder_dir):
    if images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg"):
        # display
        image_arr.append(images)


my_canvas = Canvas(root, width=800, height=600, highlightthickness=0)
my_canvas.pack()


im = Image.open(f"Images/{CURRENT_FOLDER}/{image_arr[COUNT]}")
resized = im.resize((800, 600), Image.Resampling.LANCZOS)
ph = ImageTk.PhotoImage(resized)

label = Label(my_canvas, image=ph, width=800, height=600)
label.config(anchor=CENTER)
label.pack()


def next_image():
    global COUNT
    if COUNT >= 50:  # count determined based off of how many images in folder
        next_img = Image.open(f"Images/please-wait.png")
        next_img_resized = next_img.resize((800, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(next_img_resized)
        label.config(image=photo_img)
        label.image = photo_img

    else:
        next_img = Image.open(f"Images/{CURRENT_FOLDER}/{image_arr[COUNT]}")
        next_img_resized = next_img.resize((800, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(next_img_resized)
        label.config(image=photo_img)
        label.image = photo_img
        COUNT += 1

    root.after(TIME_BETWEEN, next_image)


next_image()
mainloop()
