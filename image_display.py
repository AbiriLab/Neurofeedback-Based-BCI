import os
from os import listdir
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import time
from PIL import Image, ImageDraw, ImageFilter
import random
import csv


class DisplayImage:

    def __init__(self, master, stage):
        self.master = master
        self.image_arr = []
        self.curr_block = stage
        self.COUNT = stage * 40
        self.pause = True
        self.folder_dir = os.getcwd() + "/Images/Composite_Images"
        self.blank_img = Image.open('Images/BlankPic.jpg')
        self.TIME_BETWEEN = 1000

        r = [1, 2, 3, 4, 5, 6, 7, 8]
        random.shuffle(r)
        self.randomized_blocks = r
        print(self.randomized_blocks)

        for i in r:
            self.CURRENT_FOLDER = "/Block" + str(i)
            for images in range(40):
                img = Image.open(f"Images/Composite_Images/Block{i}/{images+1}.jpg")
                self.image_arr.append(img)
        self.my_canvas = Canvas(self.master, width=800, height=600, highlightthickness=0)
        self.my_canvas.pack()
        im = self.blank_img
        resized = im.resize((800, 600), Image.Resampling.LANCZOS)
        ph = ImageTk.PhotoImage(resized)
        self.label = Label(self.my_canvas, image=ph, width=800, height=600)
        self.label.config(anchor=CENTER)
        self.label.pack()

    def next_image(self):
        if self.pause:  # count to be determined based off of how many images in folder
            next_img = Image.open(f"Images/please-wait.png")
            next_img_resized = next_img.resize((800, 600), Image.Resampling.LANCZOS)
            photo_img = ImageTk.PhotoImage(next_img_resized)
            self.label.config(image=photo_img)
            self.label.image = photo_img
            self.master.after(self.TIME_BETWEEN, self.next_image)
        # elif self.COUNT % 2 == 0:
        else:
            next_img = self.image_arr[self.COUNT]
            next_img_resized = next_img.resize((800, 600), Image.Resampling.LANCZOS)
            photo_img = ImageTk.PhotoImage(next_img_resized)
            self.label.config(image=photo_img)
            self.label.image = photo_img
            self.COUNT += 1
            if self.COUNT % 40 == 0:
                self.pause = True
                self.curr_block += 1
            self.master.after(self.TIME_BETWEEN, self.next_image)
        # else:
        #    next_img = self.blank_img
        #    next_img_resized = next_img.resize((800, 600), Image.Resampling.LANCZOS)
        #    photo_img = ImageTk.PhotoImage(next_img_resized)
        #    self.label.config(image=photo_img)
        #    self.label.image = photo_img
        #    self.COUNT += 1
        #    if self.COUNT%80 == 0:
        #        self.pause = True
        #        self.curr_block += 1
        #    self.master.after(int(self.TIME_BETWEEN/10), self.next_image)


if __name__ == "__main__":
    root = Tk()
    root.geometry("%dx%d+%d+%d" % (800, 600, 300, 300))
    root.title("Image Slideshow")
    main_window = DisplayImage(root)
    main_window.next_image()
    root.mainloop()
