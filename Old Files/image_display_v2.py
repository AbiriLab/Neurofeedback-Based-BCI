import os
import time
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from PIL import Image, ImageDraw, ImageFilter
import random
import threading


class DisplayImage:

    def __init__(self, master, block, block_order):

        # self.top = Toplevel()
        # self.top.geometry("%dx%d+%d+%d" % (800, 600, 950, 200))
        # self.top.title("Image Window")
        self.block_order = block_order
        self.master = master
        self.image_arr = []
        self.curr_block = block
        self.COUNT = block * 40
        self.pause = True
        self.folder_dir = os.getcwd() + "/Images/Composite_Images"
        self.blank_img = Image.open('Images/BlankPic.jpg')
        self.TIME_BETWEEN = 1000
        self.randomized_blocks = None
        self.instruction_image = False
        self.single_block = False

        im = self.blank_img
        resized = im.resize((800, 600), Image.Resampling.LANCZOS)
        ph = ImageTk.PhotoImage(resized)

        self.label = Label(self.master, image=ph, width=800, height=600)
        self.label.image = ph  # need to keep the reference of your image to avoid garbage collection
        self.label.grid(row=1, column=0)

    def create_img_arr(self):
        if not self.single_block:
            block_instruct_dict = {'1': "female", '2': "female", '3': "male", '4': "male", '5': "outdoor", '6': "outdoor", '7': "indoor", '8': "indoor"}
            self.instruct_order = list(map(block_instruct_dict.get, self.block_order))
            print(self.instruct_order)
            self.p = 0

            for i in self.block_order:
                for images in range(40):
                    img = Image.open(f"Images/Composite_Images/Block{i}/{images + 1}.jpg")
                    self.image_arr.append(img)
        else:
            for n in range(40):
                img = Image.open(f"Images/Composite_Images/Block{self.block_order}/{n + 1}.jpg")
                self.image_arr.append(img)

    def next_image(self):
        if self.pause:  # count to be determined based off of how many images in folder
            next_img = Image.open(f"Images/please-wait.png")
            next_img_resized = next_img.resize((800, 600), Image.Resampling.LANCZOS)
            photo_img = ImageTk.PhotoImage(next_img_resized)
            self.label.config(image=photo_img)
            self.label.image = photo_img
            # self.label.after(self.TIME_BETWEEN, self.next_image)
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

    def close_window(self):
        self.master.destroy()

    def instructions_image(self):  # prob pass in block number( or self.randomized _blocks) and then show the image associated with it so when self.pause is true could show this image instead
        instruct_order = self.instruct_order
        target = instruct_order[self.curr_block]
        print(self.p)
        print(self.curr_block)
        instruction_img = Image.open(f"Images/Instruction Images/instruction_{target}.png")
        inst_img_resized = instruction_img.resize((800, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(inst_img_resized)
        self.label.config(image=photo_img)
        self.label.image = photo_img
        self.instruction_image = True
        self.p += 1
        self.master.after(5000, self.next_image)
