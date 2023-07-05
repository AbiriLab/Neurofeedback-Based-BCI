import os
import time
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from PIL import Image, ImageDraw, ImageFilter
import random
import threading
import numpy as np
from matplotlib import pyplot as plt


class DisplayImage:

    def __init__(self, master, block, block_order):

        self.ffaces = []
        self.sscenes = []

        folder_dir = os.getcwd() + "/Images/1-Neurofeedback/Face"
        print('ffolder_dir', folder_dir)
        for image in os.listdir(folder_dir):
            self.ffaces.append(folder_dir + "/" + image)

        folder_dir = os.getcwd() + "/Images/1-Neurofeedback/Scene"
        print('sfolder_dir', folder_dir)
        for image in os.listdir(folder_dir):
            self.sscenes.append(folder_dir + "/" + image)

        self.face_img = None
        self.scene_img = None
        
        self.image_arr = []
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

    def create_instruct_order(self):
        if not self.single_block:
            block_instruct_dict = {'1': "Face", '2': "Scene", '3': "Face", '4': "Scene", '5': "Face", '6': "Scene", '7': "Face", '8': "Scene"}
            self.instruct_order = list(map(block_instruct_dict.get, self.block_order))
            print('self.instruct_order', self.instruct_order)

    def create_img_arr(self):   
        print('Creating array')
        # Create the composite image
        composite_img = self.create_composite_img(128)  # start with equal transparency
        self.image_arr.append(composite_img)

    def create_composite_img(self, face_alpha):
        if self.face_img is None or self.scene_img is None:
            # Randomly select a face image
            face_img_path = random.choice(self.ffaces)
            self.face_img = Image.open(face_img_path).convert("L")

            # Randomly select a scene image
            scene_img_path = random.choice(self.sscenes)
            self.scene_img = Image.open(scene_img_path).convert("L")

        # Create masks with different transparencies
        face_mask = Image.new("L", self.face_img.size, int(face_alpha))
    
        # Composite greyscale images using masks
        composite_img = Image.composite(self.face_img, self.scene_img, face_mask)
        
        return composite_img 

    def update_transparency(self, face_alpha):
        # Create a new composite image with the updated transparency
        composite_img = self.create_composite_img(face_alpha)
        # Replace the last image in the array with the new composite image
        self.image_arr[-1] = composite_img
        # Update the image on the display
        self.next_image(composite_img)


        
    def next_image(self, img=None, face_alpha=.5):
        next_img = img if img is not None else self.create_composite_img(face_alpha)
        # Resize and convert to PhotoImage
        next_img_resized = next_img.resize((800, 600), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(next_img_resized)
        # Update the label
        self.label.config(image=photo_img)
        self.label.image = photo_img
        # Increment the counter
        self.COUNT = self.COUNT + 1
        # Update the master
        self.master.update()
            
    def start_new_trial(self):
        # Clear current face and scene images
        self.face_img = None
        self.scene_img = None
        # Start a new trial with a completely new image
        # You could set the initial transparency to a value of your choice, here I used 128 as in your example
        self.next_image(self.create_composite_img(128))

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