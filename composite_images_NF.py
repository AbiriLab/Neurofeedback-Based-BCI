import os
from os import listdir
from PIL import Image, ImageDraw, ImageFilter
import random
import pandas as pd


Block = '6'

# For faces
ffaces = []
folder_dir = os.getcwd() + "/Images/1-Neurofeedback/Face"
print('ffolder_dir', folder_dir)
for image in os.listdir(folder_dir):
    ffaces.append(folder_dir + "/" + image)

# For scenes
sscenes = []
folder_dir = os.getcwd() + "/Images/1-Neurofeedback/Scene"
print('sfolder_dir', folder_dir)
for image in os.listdir(folder_dir):
    sscenes.append(folder_dir + "/" + image)



# face blocks
m=None
for n in range(6):
    random.shuffle(ffaces)
    random.shuffle(sscenes)
    Block = str(n+1)
    instructions = []
    faces = []
    scenes = []
    fcount = 0
    scount = 0
    
    composite_images = []
    for i in range(40) :
        if n==0 or n==2 or n==4:
            m='Face'
        if n==1 or n==3 or n==5:
            m='Scene'
        
        instructions.append(m)
        fcount = fcount + 1
        f_img = Image.open(ffaces[fcount]).convert('L')
        faces.append('F')
        
        scount = scount + 1 
        s_img = Image.open(sscenes[scount]).convert('L')
        scenes.append('S')
        
        # Create masks with different transparencies
        very_low_mask_f = Image.new("L", f_img.size, 255)
        low_mask_f = Image.new("L", f_img.size, 192)
        normal_mask_f = Image.new("L", f_img.size, 128)
        good_mask_f = Image.new("L", f_img.size, 64)
        very_good_mask_f = Image.new("L", f_img.size, 0)
     
        mask = Image.new("L", f_img.size, 128)
        im = Image.composite(f_img, s_img, good_mask_f) # composite greyscale images using mask
        composite_images.append(im)
        
    temp = list(zip(composite_images, faces, scenes))
    random.shuffle(temp)
    s_composite_images, s_faces, s_scenes = zip(*temp)
    s_composite_images, s_faces, s_scenes = list(s_composite_images), list(s_faces), list(s_scenes)
    
    save_dir = os.path.join('Images', '1-Neurofeedback', 'Composite_Images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Continue with the code:
    for i in range(len(s_composite_images)):
        s_composite_images[i].save(os.path.join(save_dir, "{}.jpg".format(i)))
    
    data = {'Instructions': instructions, 'Face': s_faces, 'Scene': s_scenes}
    image_key = pd.DataFrame(data)
    image_key.to_csv('Images/1-Neurofeedback/Composite_Images' + Block + '_key.csv')
    del instructions
    del faces
    del scenes
    del composite_images
    del s_faces
    del s_scenes
    del s_composite_images
    del data
    del image_key
    del temp



