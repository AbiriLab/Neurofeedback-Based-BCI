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

        self.block_order = block_order
        self.master = master
        self.image_arr = []
        self.curr_block = block
        self.COUNT = 0
        self.pause = True
        self.folder_dir = os.getcwd() + "/Images/1-Neurofeedback"
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
            block_instruct_dict = {'1': "Face", '2': "Scene", '3': "Face", '4': "Scene", '5': "Face", '6': "Scene"}
            self.instruct_order = list(map(block_instruct_dict.get, self.block_order))
            print('self.instruct_order', self.instruct_order)
            self.p = 0
            print('Creating array')
            for i in self.block_order:
                base_dir = "C:/Users/tnlab/OneDrive/Documents/GitHub/Neurofeedback-Based-BCI/Images/1-Neurofeedback/Composite_Images/"
                for images in range(40):
                    image_path = os.path.join(base_dir, f"{images}.jpg")
                    img = Image.open(image_path)
                    self.image_arr.append(img)


    def next_image(self):
        next_img = self.image_arr[self.COUNT]
        next_img_resized = next_img.resize((800, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(next_img_resized)
        self.label.config(image=photo_img)
        self.label.image = photo_img
        self.COUNT = self.COUNT + 1
        self.master.update()

    def close_window(self):
        self.master.destroy()

    def instructions_image(self):  # prob pass in block number( or self.randomized _blocks) and then show the image associated with it so when self.pause is true could show this image instead
        base_dir_ins = "C:/Users/tnlab/OneDrive/Documents/GitHub/Neurofeedback-Based-BCI/Images/1-Neurofeedback/Neurofeedback_Instruction"
        instruct_order = self.instruct_order
        target = instruct_order[self.curr_block]
        instruction_img = Image.open(os.path.join(base_dir_ins, f"instruction_{target}.png"))
        inst_img_resized = instruction_img.resize((800, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(inst_img_resized)
        self.label.config(image=photo_img)
        self.label.image = photo_img
        self.curr_block = self.curr_block + 1

    def pleaseWait_image(self):
        next_img = Image.open(f"Images/please-wait.png")   
        next_img_resized = next_img.resize((800, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(next_img_resized)
        self.label.config(image=photo_img)
        self.label.image = photo_img