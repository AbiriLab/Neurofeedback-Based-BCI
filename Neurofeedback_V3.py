from tkinter import ttk, Tk
from tkinter import *
import csv
import pandas as pd
import numpy as np
from image_display_unicorn import *
from image_display_unicorn_NF import *
import UnicornPy
import random
from scipy.signal import butter, filtfilt     
####################################################
import os
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from sklearn.utils import shuffle
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Dense,  BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
##############
from sklearn.svm import SVC
from collections import Counter
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import hilbert
from scipy import stats
##################################
from os import listdir
from PIL import Image, ImageDraw, ImageFilter,ImageTk
import threading
import time
import subprocess
from screeninfo import get_monitors
###############################################################################################
from numpy import unwrap, diff, abs, angle
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from sklearn.utils import shuffle
import scipy
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Dense,  BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import mne
from mne.preprocessing import ICA

device = UnicornPy.Unicorn("UN-2021.05.36")


# KeyPressDetector class
class KeyPressDetector:
    def __init__(self, master):
        self.key_pressed = 0  # Initial state
        master.bind('<Key>', self.on_key_press)
    def on_key_press(self, event=None):
        print("Key pressed detected!")
        self.key_pressed = 1
    def check_key_press(self):
        result = self.key_pressed
        self.key_pressed = 0  # Reset after checking
        return result

class RootWindow:
    def __init__(self, master):
        self.master = master  
        self.SamplingRate = UnicornPy.SamplingRate
        self.SerialNumber = 'UN-2021.05.36'
        self.numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()
        self.configuration = device.GetConfiguration()
        self.AcquisitionDurationInSeconds = 1 
        self.FrameLength=1
        self.numberOfGetDataCalls = int(self.AcquisitionDurationInSeconds * self.SamplingRate / self.FrameLength)
        self.receiveBufferBufferLength = self.FrameLength * self.numberOfAcquiredChannels*4   # 4 bytes per float32
        self.receiveBuffer = bytearray(self.receiveBufferBufferLength)
        self.eeg_data = []
        self.image_window = None
        self.block = 0
        self.image_window_open = False
        self.patient_progress = ['', '0', '0', '0', '00000000','00000000','00000000','00000000']
        self.patient_index = 0
        self.patient_data_list = []
        self.pre_eval = 0
        self.neuro = 0
        self.post_eval = 0
        self.seq = None
        self.top = None
        self.r = [1, 2, 3, 4, 5, 6, 7, 8]
        self.instruction_mapping = {1: 'Face', 2: 'Scene', 3: 'Face', 4: 'Scene', 5: 'Face', 6: 'Scene', 7: 'Face', 8: 'Scene'}
        self.create_gui(master)
        self.key_detector = KeyPressDetector(master)

        with open('pat_progess_v2.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.patient_data_list.append(row)
        df = pd.read_csv('pat_progess_v2.csv')
    
    def create_gui(self, master):
        master.geometry("1250x450+100+50")
        master.resizable(True, True)
        self.frame_1 = tk.Frame(master)
        self.frame_1.pack()
        
        self.patient_name_label = tk.Label(self.frame_1, text="Patient Name:", font=15)
        self.patient_name_label.grid(row=1, column=0, pady=15, padx=5)
        
        self.patient_name_data = tk.StringVar()
        self.patient_name_entry = tk.Entry(self.frame_1, width=30, font=15, textvariable=self.patient_name_data)
        self.patient_name_entry.grid(row=1, column=1, pady=15, padx=5)
        
        self.submit_button = tk.Button(self.frame_1, text="Create Folder", command=self.get_patient_name_and_create_folder)
        self.submit_button.grid(row=1, column=2, pady=15, padx=5)   
        
        self.phase_label = tk.Label(self.frame_1, text="Current Phase:", font=15)
        self.phase_label.grid(row=4, column=0, pady=15, padx=5)
        self.curr_phase = tk.StringVar()
        self.curr_phase_num = tk.Label(self.frame_1, textvariable=self.curr_phase, font=15)
        self.curr_phase_num.grid(row=4, column=1, pady=15, padx=5)
        plist = ["Pre-Evaluation", "Neurofeedback", "Post-Evaluation"]
        self.phase_box = ttk.Combobox(self.frame_1, values=plist, state='readonly', font=15, textvariable=self.curr_phase)
        self.phase_box.set("Select the Phase")
        self.phase_box.grid(row=3, column=1, pady=15, padx=5)

        self.block_label = tk.Label(self.frame_1, text="Last Block\n Completed:", font=10)
        self.block_label.grid(row=5, column=0, pady=15, padx=5)
        self.curr_block = StringVar()
        self.block_num = tk.Label(self.frame_1, textvariable=self.curr_block, font=15)
        self.block_num.grid(row=5, column=1, columnspan=2, pady=15, padx=5)

        self.phase_prog_lab = tk.Label(self.frame_1, text="Phase Progress:", font=15)
        self.phase_prog_lab.grid(row=6, column=0, pady=15, padx=5)
        self.progress = tk.IntVar()
        self.phase_prog = ttk.Progressbar(self.frame_1, variable=self.progress, length=250)
        self.progress.set(0)
        self.phase_prog.grid(row=6, column=1, pady=15, padx=5)

        self.frame_1.grid(padx=30, pady=50, row=0, column=0)

        self.frame_2 = tk.Frame(master)

        self.create_trial_but = Button(self.frame_2, text="Create Trial", bg="green", font=("TkDefaultFont", 15), command=self.create_trial)
        self.create_trial_but.grid(row=1, column=0, columnspan=3, pady=20, padx=5)

        self.start_trial_but = Button(self.frame_2, text="Start Trial", bg="green",  font=("TkDefaultFont", 15), width=20, command=self.start_trial_thread) #self.start_trial                           
        self.start_trial_but.grid(row=4, column=0, columnspan=4, pady=8, padx=5)   

        self.start_trial_but = Button(self.frame_2, text="Create Classifier", bg="yellow", font=10, command=self.Create_SVMScript) #self.start_trial                           
        self.start_trial_but.grid(row=5, column=0, pady=15, padx=5) 
        
        self.create_trial_but = Button(self.frame_2, text="Recalibrate SVM", bg="yellow", font=10, command=self.run_RSVMScript)
        self.create_trial_but.grid(row=5, column=1, columnspan=2, pady=15, padx=5)
        
        self.end_trial_but = Button(self.frame_2, text="Score", bg="pink", font=("TkDefaultFont", 15), width=20, command=self.HBScore)
        self.end_trial_but.grid(row=6, column=0, columnspan=4, pady=20, padx=5)    

        self.end_trial_but = Button(self.frame_2, text="End Trial", bg="red", font=("TkDefaultFont", 15), width=20, command=self.end_trial)
        self.end_trial_but.grid(row=7, column=0, columnspan=4, pady=20, padx=5)
        
        self.frame_2.grid(padx=30, pady=50, row=0, column=1)

        self.frame_3 = tk.Frame(master)

        self.single_block = tk.Label(self.frame_3, text="Single Block", font=8)
        self.single_block.grid(row=2, column=0, pady=5, padx=3)
        blist = ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.single_block_num_var = IntVar()
        self.block_box = ttk.Combobox(self.frame_3, values=blist, state='readonly', font=10,
                                      textvariable=self.single_block_num_var, width=12)
        self.block_box.set("Block to Run")
        self.block_box.grid(row=2, column=1, pady=5, padx=3)

        self.start_block_but = Button(self.frame_3, text="Start Block", bg="light green", font=8, command=self.start_block)
        self.start_block_but.grid(row=3, column=0, pady=5, padx=3)

        self.end_block_but = Button(self.frame_3, text="End Block", bg='#FB586C', font=8, command=self.end_block)
        self.end_block_but.grid(row=3, column=1, pady=5, padx=3)

        self.frame_3.grid(padx=30, pady=50, row=0, column=2)
    
    def update_progress(self, progress):
        scaled_progress = (progress / 6) * 100 # Scale progress to match the range of the progress bar (0-100)
        self.progress.set(scaled_progress)
    
    def get_patient_name_and_create_folder(self):
        patient_name = self.patient_name_data.get()
        if patient_name:
            self.create_patient_folder(patient_name)
        else:
            print("Please enter a patient name.")    
   
    def create_patient_folder(self, patient_name):
        root_folder = "2-Patient Data"
        patient_folder = os.path.join(root_folder, patient_name)
        try:
            os.makedirs(patient_folder, exist_ok=True)
            print(f"Folder created: {patient_folder}")
            sub_folders = ["Pre Evaluation", "Post Evaluation", "Neurofeedback"]
            for sub_folder in sub_folders:
                sub_folder_path = os.path.join(patient_folder, sub_folder)
                os.makedirs(sub_folder_path, exist_ok=True)
                print(f"Sub-folder created: {sub_folder_path}")
        except Exception as e:
            print(f"An error occurred while creating the folder: {e}")
           
    def HBScore(self):
        phase = self.curr_phase.get()
        if phase == "Pre-Evaluation":
            folder_name = "Pre Evaluation"
        elif phase == "Post-Evaluation":
            folder_name = "Post Evaluation"
        else:
            print("Phase not set.")
            return
        patient_name = self.patient_name_data.get()
        if not patient_name:
            print("Patient name not set.")
            return
        image_path = os.path.join("2-Patient Data", patient_name, folder_name, "Score.png")
        if not os.path.exists(image_path):
            print("Image file does not exist!")
            return
        pil_image = Image.open(image_path)
        pil_image.show()
           
    # def create_trial(self):
    #     print('self.patient_progress', len(self.patient_progress))
    #     patient_name = self.patient_name_entry.get()
    #     self.patient_progress[0] = patient_name
    #     random.shuffle(self.r)
    #     randomized_blocks = self.r
    #     self.seq = " ".join(str(x) for x in randomized_blocks)
    #     print('self.seq', self.seq) 
    #     print('randomized_blocks', randomized_blocks)
    #     # for data_list in self.patient_data_list:
    #     #    if patient_name == data_list[0]:
    #     # self.patient_index = self.patient_data_list.index(data_list)
    #     # self.patient_progress = data_list
    #     self.pre_eval = int(self.patient_progress[1])
    #     self.neuro = int(self.patient_progress[2])
    #     self.post_eval = int(self.patient_progress[3])
    #     # self.seq = self.patient_progress[4]
    #     if self.curr_phase.get() == 'Pre-Evaluation':
    #         self.block = self.pre_eval
    #     elif self.curr_phase.get() == "Neurofeedback":
    #         self.block = self.neuro
    #     elif self.curr_phase.get() == "Post-Evaluation":
    #         self.block = self.post_eval
    #     self.curr_block.set(str(self.block))
    #     print('self.curr_block', self.curr_block)
    #     self.progress.set(round(self.block * 12.5))
    #     # self.open_image_win()
    #     self.start_trial_but.config(state="normal")
    #     # return
    #     # self.pre_eval, self.neuro, self.post_eval = 0, 0, 0
    #     if self.curr_phase.get() == 'Pre-Evaluation':
    #         self.patient_progress[5] = self.seq
    #     elif self.curr_phase.get() == "Neurofeedback":
    #         self.patient_progress[6] = self.seq
    #     elif self.curr_phase.get() == "Post-Evaluation":
    #         self.patient_progress[7] = self.seq
           
    #     self.patient_progress[4] = self.seq
    #     self.patient_data_list.append(self.patient_progress)
    #     self.patient_index = len(self.patient_data_list) - 1
    #     self.add_patient_data()
    #     self.open_image_win()
    #     # self.update_gui()
    

    def create_trial(self):
        print('self.patient_progress', len(self.patient_progress))
        patient_name = self.patient_name_entry.get()
        self.patient_progress[0] = patient_name
        random.shuffle(self.r)
        randomized_blocks = self.r
        self.seq = " ".join(str(x) for x in randomized_blocks)
        print('self.seq', self.seq) 
        print('randomized_blocks', randomized_blocks)
        
        self.pre_eval = int(self.patient_progress[1])
        self.neuro = int(self.patient_progress[2])
        self.post_eval = int(self.patient_progress[3])

        if self.curr_phase.get() == 'Pre-Evaluation':
            self.block = self.pre_eval
        elif self.curr_phase.get() == "Neurofeedback":
            self.block = self.neuro
        elif self.curr_phase.get() == "Post-Evaluation":
            self.block = self.post_eval
        self.curr_block.set(str(self.block))
        print('self.curr_block', self.curr_block)
        self.progress.set(round(self.block * 12.5))

        if self.curr_phase.get() == 'Pre-Evaluation':
            self.patient_progress[5] = self.seq
        elif self.curr_phase.get() == "Neurofeedback":
            self.patient_progress[6] = self.seq
        elif self.curr_phase.get() == "Post-Evaluation":
            self.patient_progress[7] = self.seq

        self.patient_progress[4] = self.seq

        # Check if patient exists in the list and update, else append new data
        patient_found = False
        for index, data_list in enumerate(self.patient_data_list):
            if patient_name == data_list[0]:
                self.patient_data_list[index] = self.patient_progress
                patient_found = True
                break
        if not patient_found:
            self.patient_data_list.append(self.patient_progress)
            self.patient_index = len(self.patient_data_list) - 1
        self.add_patient_data()
        self.open_image_win()
   
    def add_patient_data(self):
        with open('pat_progess_v2.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in self.patient_data_list:
                writer.writerow(row)

    # def get_second_monitor_geometry():
    #     monitors = get_monitors()
    #     if len(monitors) < 2:
    #         print('Just one monitor has been detected')
    #         return None
    #     second_monitor = monitors[1]  # assuming the second monitor is the second in the list
    #     width = second_monitor.width
    #     height = second_monitor.height
    #     x = second_monitor.x
    #     y = second_monitor.y
    #     print('The second monitor has been detected')
    #     # return the center position for an 800x600 window
    #     return "%dx%d+%d+%d" % (1600, 1200, x + (width - 1600) // 2, y + (height - 1200) // 2)

    def open_image_win(self):
        global image_window, top
        top = Toplevel()
        monitors = get_monitors()
        if len(monitors) >= 2:
            print('Two monitors have been detected')
            monitor = monitors[1]  # This is the second monitor
            # Set the window to be in the middle of the second monitor
            # Double the window dimensions
            window_width = monitor.width
            window_height = monitor.height
            x = monitor.x #- (monitor.x) // 2
            y = monitor.y #+ (monitor.height - window_height) // 2
            top.geometry("%dx%d+%d+%d" % (window_width, window_height, x, y))
        else:
            # Fallback for single monitor setups
            top.geometry("%dx%d+%d+%d" % (1600, 1200, 300, 200))
        top.title("Image Slideshow")

        if self.curr_phase.get() == "Neurofeedback":  
            image_window = DisplayImagenf(top, self.block, list(self.seq.replace(" ", "")))
            image_window.single_block = False
            image_window.create_instruct_order() 
            image_window.create_img_arr_nf()
            image_window.pleaseWait_image()
            self.image_window_open = True    
        else:
            image_window = DisplayImage(top, self.block, list(self.seq.replace(" ", "")))
            image_window.single_block = False
            image_window.create_img_arr()
            image_window.pleaseWait_image()
            self.image_window_open = True
        top.update()
    ################################################################################################################################    
    ################################################################################################################################
    ################################################################################################################################   
    #Neurofeedback 
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5, initial_state=None):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        # If no initial state is provided, calculate it.
        if initial_state is None:
            zi = lfilter_zi(b, a)
            initial_state = zi * data[0]
        y, final_state = lfilter(b, a, data, zi=initial_state)
        return y, final_state

    def denoise_data(self, df, col_names, n_clusters):
        df_denoised = df.copy()
        for col_name, k in zip(col_names, n_clusters):
            df_denoised[col_name] = pd.to_numeric(df_denoised[col_name], errors='coerce') # Convert column to numeric format
            X = df_denoised.select_dtypes(include=['float64', 'int64']) # Select only numeric columns
            clf = KNeighborsRegressor(n_neighbors=k, weights='uniform') # Fit KNeighborsRegressor
            clf.fit(X.index.values[:, np.newaxis], X[col_name])
            y_pred = clf.predict(X.index.values[:, np.newaxis]) # Predict values 
            df_denoised[col_name] = y_pred
        return df_denoised

    def z_score(self, df, col_names):
        df_standard = df.copy()
        for col in col_names:
            df_standard[col] = (df[col] - df[col].mean()) / df[col].std()
        return df_standard

    def custom_detrend(self, df, col_names):
        df_detrended = df.copy()
        for col in col_names:
            y = df_detrended[col]
            x = np.arange(len(y))
            p = np.polyfit(x, y, 1)
            trend = np.polyval(p, x)
            detrended = y - trend
            df_detrended[col] = detrended
        return df_detrended

    def preprocess(self, df, col_names, n_clusters):
        df_new = df.copy()
        df_new = self.denoise_data(df, col_names, n_clusters)
        df_new = self.z_score(df_new, col_names)
        df_new = self.custom_detrend(df_new, col_names)
        return df_new

    def reject_artifacts(self, df, channel):
        threshold_factor =3
        median = df[channel].median()
        mad = np.median(np.abs(df[channel] - median))
        spikes = np.abs(df[channel] - median) > threshold_factor * mad
        x = np.arange(len(df[channel]))
        cs = CubicSpline(x[~spikes], df[channel][~spikes])    # Interpolate using Cubic Spline
        interpolated_values = cs(x)
        interpolated_values[spikes] *= 0.1  # Make interpolated values 0.001 times smaller
        df[channel] = interpolated_values
        return df

    # def reject_artifacts_DN(self, df, channel):
    #     threshold_factor =5
    #     median = df[channel].median()
    #     mad = np.median(np.abs(df[channel] - median))
    #     spikes = np.abs(df[channel] - median) > threshold_factor * mad
    #     x = np.arange(len(df[channel]))
    #     cs = CubicSpline(x[~spikes], df[channel][~spikes])    # Interpolate using Cubic Spline
    #     interpolated_values = cs(x)
    #     interpolated_values[spikes] *= 0.5  # Make interpolated values 0.001 times smaller
    #     df[channel] = interpolated_values
    #     return df

    def reject_artifacts(df, channel):
        threshold_factor = 3
        median = df[channel].median()
        mad = np.median(np.abs(df[channel] - median))
        spikes = np.abs(df[channel] - median) > threshold_factor * mad
        x = np.arange(len(df[channel]))
        cs = CubicSpline(x[~spikes], df[channel][~spikes]) # Interpolate using Cubic Spline
        interpolated_values = cs(x)
        interpolated_values[spikes] *= 0.01  # Make interpolated values 0.1 times smaller
        # Check each interpolated value's difference from median and compare to the threshold
        spike_values = np.abs(interpolated_values - median) > threshold_factor * mad
        interpolated_values[spike_values] *= 0.01 
        spike_values = np.abs(interpolated_values - median) > threshold_factor * mad
        interpolated_values[spike_values] *= 0.01 
        df[channel] = interpolated_values
        return df


    def apply_bandpass_filter(self, signal, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

                
    def start_trial_thread(self):
        t = threading.Thread(target=self.start_trial)
        t.start()   
          
    def start_trial(self):
        global image_window, top
        self.master.focus_set()
        patient_name = self.patient_name_data.get() # Get patient name and create the patient folder
        patient_folder= os.path.join("2-Patient Data", patient_name)
        pre_folder= os.path.join(patient_folder, "Pre Evaluation")
        post_folder= os.path.join(patient_folder, "Post Evaluation")
        neuro_folder= os.path.join(patient_folder, "Neurofeedback")
        frequency_bands = {'delta': (0.5, 4),'theta': (4, 8),'alpha': (8, 14),'beta': (14, 30),'gamma': (30, 40),'ERP':(0.4,40)}
        fs=250
      
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)
        
        if self.curr_phase.get() == "Neurofeedback":  
            seq_list = [int(x) for x in self.seq if x.isdigit()]
            print('seq_list', seq_list)
            print('Trial',self.block)
            self.patient_progress[2]=self.block+1
            print('randomized_blocks:',seq_list[self.block])
            
            
            device.StartAcquisition(False)
            image_window.instructions_image_nf()
            top.update()
            time.sleep(5)
            
        
            filename = r'C:\Users\tnlab\OneDrive\Documents\GitHub\AlphaFold\Neurofeedback-Based-BCI\my_svm_model.joblib'
            svm_model = joblib.load(filename)
            
            # Initialize the buffer
            buffer_size_seconds = 5
            samples_per_second = 250 
            buffer_size_samples = buffer_size_seconds * samples_per_second
            buffer = np.zeros((buffer_size_samples, 8))  # 8 is the number of EEG channels
            face_alpha_values = [0,70,128,204,255] 
            face_alpha_index=2
            
            current_directory = os.getcwd()
            print(f"Current directory: {current_directory}")
            
            final_lable_array=[]
            raw=[]
            PP=[]
            for j in range (0,8):
                image_window.start_new_trial()
                selected_columns = ['Fz', 'FC1', 'FC2', 'C3', 'Cz', 'C4', 'CPz','Pz']
                tdata=[]
                lable=[]
                for n in range(0,5): #looking at each image for 5 seconds
                    print('n', n)
                    for i in range(self.numberOfGetDataCalls): 
                        device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                        dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                        data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength
                        tdata.append(data.copy())
                        tdataarray=np.array(tdata)    
                    new_totdata_array = tdataarray.reshape(-1, 17) 
                    # print('new_totdata_array',type(new_totdata_array), new_totdata_array.shape ) #'numpy.ndarray', (250, 17--1250,17)
                    # inst = new_totdata_array[:, 16]
                    # event.append(inst)
                    Last_data=new_totdata_array[:, :8]
                    # Last_data.columns = selected_columns 
                    raw.append(Last_data)

                    buffer = np.append(buffer, Last_data[-250:, :], axis=0)
                    if buffer.shape[0] > buffer_size_samples:
                        num_extra_samples = buffer.shape[0] - buffer_size_samples
                        buffer = buffer[num_extra_samples:, :]
                    df = pd.DataFrame(buffer)
                    Combined_raw_eeg_nf_bp = np.copy(buffer)
                    
                    num_columns_nf = buffer.shape[1]
                    filter_states = [None] * num_columns_nf  # Initialize a list to hold states for each column
                    for column in range(num_columns_nf):
                        Combined_raw_eeg_nf_bp[:, column], filter_states[column] = self.butter_bandpass_filter(
                            Combined_raw_eeg_nf_bp[:, column], lowcut=.4, highcut=40, fs=250, order=5, initial_state=filter_states[column])
                    combined_raw_eeg_nf_bp = pd.DataFrame(Combined_raw_eeg_nf_bp)
                    # combined_raw_eeg_nf_bp.to_csv(f"bufferbp_{j}_{n}.csv", index=False)
                    # print('combined_raw_eeg_nf_bp', type(combined_raw_eeg_nf_bp), combined_raw_eeg_nf_bp.shape)
                    
                    # 2. Artifact rejection
                    BP_artifact_RJ = combined_raw_eeg_nf_bp.copy()
                    initial_BP_artifact_RJ = BP_artifact_RJ.iloc[:-(n+1)*250]
                    # print('initial_BP_artifact_RJ', initial_BP_artifact_RJ.shape)
                    # print('BP_artifact_RJ ', type(BP_artifact_RJ ))
                    for channel in range (8):
                        BP_artifact_RJ= self.reject_artifacts(BP_artifact_RJ.iloc[-(n+1)*250:], channel)
                    # print('BP_artifact_RJ', type(BP_artifact_RJ), BP_artifact_RJ.shape)
                    
                    # 3. Smoothing
                    BP_artifact_RJ_SM=BP_artifact_RJ.copy()
                    window_size = 10 
                    for channel in range (8):
                        BP_artifact_RJ_SM= BP_artifact_RJ_SM.rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')                
                    # print('BP_artifact_RJ_SM', type(BP_artifact_RJ_SM), BP_artifact_RJ_SM.shape)
                    DN=pd.concat([initial_BP_artifact_RJ, BP_artifact_RJ_SM], axis=0)
                    # print('DN', DN.shape, DN)
                    # 4. Denoising and other preprocessing
                    DN.columns = selected_columns
                    eeg_df_denoised = self.preprocess(DN, col_names=selected_columns, n_clusters=[50]*len(selected_columns))
                    # print('eeg_df_denoised', type(eeg_df_denoised), eeg_df_denoised, eeg_df_denoised.shape )
                            
                    chunks = np.array_split(eeg_df_denoised.to_numpy(), 5, axis=0)
                    # print('chunks', chunks)
                    
                    # print('chunks[4]',chunks[4].shape)
                    eeg_signal = chunks[4].reshape(8, 250)  # reshaped to (8, 250)
                    # print('eeg_signal', eeg_signal.shape)
                    Hil_feature_for_sample = []
                    Power_feature_for_sample = []
                    for channel in range(8):
                        channel_signal = eeg_signal[channel, :]
                        hilbert_for_channel=[]
                        power_for_channel = []
                        for band, (low, high) in frequency_bands.items():
                            filtered_signal = self.apply_bandpass_filter(channel_signal, low, high, fs)
                            hilbert_features = self.calculate_hilbert_features(filtered_signal, fs)
                            mean_amplitude = np.mean((filtered_signal)**2)
                            power_for_channel.append(mean_amplitude)
                            # print('hilbert_features', hilbert_features)
                            hilbert_for_channel.append(hilbert_features) 
                        Hil_feature_for_sample.append(hilbert_for_channel)
                        Power_feature_for_sample.append(power_for_channel)
                    BP_Power_FE_np = np.array(Power_feature_for_sample)     
                    Hil_FE_np=np.array(Hil_feature_for_sample)
                    # print('Hil_FE_np', Hil_FE_np.shape)
                    
                    ERP_FE = np.array([self.extract_ERP_features(eeg_signal[j, :]) for j in range(8)])
                    print(ERP_FE.shape)
                    combined_features = np.concatenate([Hil_FE_np, BP_Power_FE_np, ERP_FE], axis=1)
                    X_n=combined_features.reshape(-1,144)
                    predictions =svm_model.predict(X_n)
                    instruction = self.instruction_mapping[seq_list[self.block]]
                    correct_prediction = (instruction == 'Face' and predictions[0] == 0) or (instruction == 'Scene' and predictions[0] == 1)

                    label_array = np.zeros((250, 4), dtype=object) 
                    label_array[:, 2] = 'F' if instruction == 'Face' else 'S'
                    
                    # for row in range(label_array.shape[0]):
                    #     if (instruction == 'Face' and correct_prediction) or (instruction == 'Scene' and not correct_prediction):
                    #         label_array[row, 1] = 'F'
                    #     elif (instruction == 'Scene' and correct_prediction) or (instruction == 'Face' and not correct_prediction):
                    #         label_array[row, 1] = 'S'
                    
                    for row in range(label_array.shape[0]):
                        if (instruction == 'Face' and correct_prediction):
                            label_array[row, 0] = 'F'
                        if (instruction == 'Scene' and correct_prediction):
                            label_array[row, 0] = 'S'
                        elif (instruction == 'Face' and not correct_prediction) or (instruction == 'Scene' and not correct_prediction):
                            label_array[row, 0] = 'N'
                    
                    label_array[:, 1] =label_array[:, 0]         
                    label_array[:, 3] = 1 if correct_prediction else 0
                    # print('label_array',label_array.shape)
        
                    # Adjust alpha
                    if instruction == 'Face':
                        if correct_prediction:
                            face_alpha_index = min(face_alpha_index + 1, len(face_alpha_values) - 1)
                            # print(f"Face mask increased in the {n}th second.", 'face alpha index=', face_alpha_index)
                        else:
                            face_alpha_index = max(face_alpha_index - 1, 0)
                            # print(f"Face mask decreased in the {n}th second.", 'face alpha index=', face_alpha_index)
                    else:  # Instruction is 'Scene'
                        if correct_prediction:
                            face_alpha_index = max(face_alpha_index - 1, 0) 
                            # print(f"Face mask decreased in the {n}th second.", 'face alpha index=', face_alpha_index)
                        else:
                            face_alpha_index = min(face_alpha_index + 1, len(face_alpha_values) - 1)
                            # print(f"Face mask increased in the {n}th second.", 'face alpha index=', face_alpha_index)
                    
                    new_face_alpha=face_alpha_values[face_alpha_index]
                    image_window.update_transparency(new_face_alpha)
                    
                    lable.append(label_array)
                    nplable=np.array(lable).reshape(-1, 4)
                    fal = np.concatenate((new_totdata_array, nplable), axis=1)
                    # print('nplable', nplable.shape)
                del tdata
                
                final_lable_array.append(fal)
                fl=np.array(final_lable_array).reshape(-1, 21)
                # print('fl', fl.shape)
            del seq_list 
            csv_filename = f'fl_{image_window.curr_block}.csv'
            csv_filepath = os.path.join(neuro_folder, csv_filename)
            
            
            with open(csv_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in fl:
                    writer.writerow(row)                    
        else: 
            seq_list = [int(x) for x in self.seq if x.isdigit()]
            print('self.seq', self.seq)
            print('seq_list', seq_list)
            print('Trial',self.block)
            if self.curr_phase.get() == 'Pre-Evaluation':
                self.patient_progress[1]=self.block+1
            else:
                self.patient_progress[3]=self.block+1
                
            print('randomized_blocks:',seq_list[self.block])
            image_window.instructions_image()
            top.update()
            device.StartAcquisition(False)
            
            instruction_duration_samples = 250 * 5
            instruction_samples_collected = 0
            while instruction_samples_collected < instruction_duration_samples:
                device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                instruction_samples_collected += self.FrameLength
            excel_file_lable = pd.read_csv(f'Block{seq_list[self.block]}_key.csv')
            
            tdataarray=[]
            tdata=[]
            root.update()
            
            for j in range (0,42):
                row_data = excel_file_lable.iloc[j,[1, 2, 3]].to_numpy()
                print('row_data', row_data)
                image_window.next_image()  
                if j==0:
                      
                    for i in range(0, 5*self.numberOfGetDataCalls): #self.numberOfGetDataCalls=250
                        device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                        key_pressed = self.key_detector.check_key_press()
                        # Convert receive buffer to numpy float array 
                        dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                        data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength,
                        combined_data = np.concatenate((data, row_data))
                        combined_data = np.concatenate((combined_data,[key_pressed]))
                        tdata.append(combined_data)
                        tdataarray = np.array(tdata)    

                    csv_filename = f'raw_eeg_block_{seq_list[self.block]}.csv'
                    if self.curr_phase.get() == "Pre-Evaluation":  
                        csv_filepath = os.path.join(pre_folder, csv_filename)
                    if self.curr_phase.get() == "Post-Evaluation":  
                        csv_filepath = os.path.join(post_folder, csv_filename)
                    with open(csv_filepath, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for row in tdataarray:
                            writer.writerow(row)

                elif j==1:
                    for i in range(0, 2*self.numberOfGetDataCalls): #self.numberOfGetDataCalls=250
                        device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                        key_pressed = self.key_detector.check_key_press()
                        # Convert receive buffer to numpy float array 
                        dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                        data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength,
                        combined_data = np.concatenate((data, row_data))
                        combined_data = np.concatenate((combined_data,[key_pressed]))
                        tdata.append(combined_data)
                        tdataarray = np.array(tdata)    
                    csv_filename = f'raw_eeg_block_{seq_list[self.block]}.csv'
                    if self.curr_phase.get() == "Pre-Evaluation":  
                        csv_filepath = os.path.join(pre_folder, csv_filename)
                    if self.curr_phase.get() == "Post-Evaluation":  
                        csv_filepath = os.path.join(post_folder, csv_filename)
                    with open(csv_filepath, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for row in tdataarray:
                            writer.writerow(row)
                else:
                    for i in range(0, self.numberOfGetDataCalls): #self.numberOfGetDataCalls=250
                        device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                        key_pressed = self.key_detector.check_key_press()
                        # Convert receive buffer to numpy float array 
                        dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                        data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength,
                        combined_data = np.concatenate((data, row_data))
                        combined_data = np.concatenate((combined_data,[key_pressed]))
                        tdata.append(combined_data)
                        tdataarray = np.array(tdata)    
                    csv_filename = f'raw_eeg_block_{seq_list[self.block]}.csv'
                    if self.curr_phase.get() == "Pre-Evaluation":  
                        csv_filepath = os.path.join(pre_folder, csv_filename)
                    if self.curr_phase.get() == "Post-Evaluation":  
                        csv_filepath = os.path.join(post_folder, csv_filename)
                    with open(csv_filepath, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for row in tdataarray:
                            writer.writerow(row)

                        
                print(j) 
            del seq_list  
            del tdata
            del tdataarray
        image_window.pleaseWait_image()   
        self.update_gui()
        self.update_patient_data() 
        device.StopAcquisition() 

    ################################################################################################################################    
    ################################################################################################################################
    ################################################################################################################################  
    def Create_SVMScript(self):
        try:
            subprocess.run(["python", "RSVM_V1.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"The RSVM_V1.py script encountered an error: {e}")
        except FileNotFoundError:
            print("The RSVM_V1.py script was not found.")

    def run_RSVMScript(self):
        try:
            subprocess.run(["python", "RSVM_V1.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"The RSVM_V1.py script encountered an error: {e}")
        except FileNotFoundError:
            print("The RSVM_V1.py script was not found.")
            
    def end_trial(self):
        global image_window
        del self.seq
        self.add_patient_data()
        image_window.close_window()
        device.StopAcquisition()
        print('Disconnected')

    def update_gui(self):
        self.block += 1
        self.curr_block.set(str(self.block))
        self.progress.set(round(self.block * 12.5))

    def start_block(self):
        global image_window
        patient_name = self.patient_name_data.get() # Get patient name and create the patient folder
        patient_folder= os.path.join("2-Patient Data", patient_name)
        pre_folder= os.path.join(patient_folder, "Pre Evaluation")
        post_folder= os.path.join(patient_folder, "Post Evaluation")
        self.master.focus_set()
        block_to_run = self.single_block_num_var.get()
        # if self.image_window_open:
        #     self.image_window.close_window()
        print('Block', block_to_run)    
        self.top = Toplevel()    
        monitors = get_monitors()
        
        if len(monitors) >= 2:
            print('Two monitors have been detected')
            monitor = monitors[1]  # This is the second monitor
            # Set the window to be in the middle of the second monitor
            # Double the window dimensions
            window_width = monitor.width
            window_height = monitor.height
            x = monitor.x #- (monitor.x) // 2
            y = monitor.y #+ (monitor.height - window_height) // 2
            self.top.geometry("%dx%d+%d+%d" % (window_width, window_height, x, y)) #(window_width, window_height, x, y)
        else:
            # Fallback for single monitor setups
            self.top.geometry("%dx%d+%d+%d" % (1600, 1200, 300, 200))
        self.top.title("Image Slideshow")
        
        # # Check if there are at least two monitors
        # if len(monitors) >= 2:
        #     monitor = monitors[1]  # This is the second monitor
        #     # Set the window to be in the middle of the second monitor
        #     x = monitor.x + (monitor.width - 800) // 2
        #     y = monitor.y + (monitor.height - 600) // 2
        #     self.top.geometry("%dx%d+%d+%d" % (800, 600, x, y))
        # else:
        #     # Fallback for single monitor setups
        #     self.top.geometry("%dx%d+%d+%d" % (800, 600, 500, 200))

        # self.top.title("Image Slideshow")
        # self.top.geometry("%dx%d+%d+%d" % (800, 600, 950, 200))
        # self.top.title("Image Slideshow")
            
        self.image_window = DisplayImage(self.top, self.block, block_to_run)
        self.image_window.single_block = True
        self.image_window.create_img_arr()
        self.image_window.pleaseWait_image()
        self.image_window_open = True
        # self.image_window.pause = False
        
        self.image_window.instructions_image() 
        self.top.update()
        self.image_window_open = True
        print("Image Window Opened") 
        device.StartAcquisition(False)
            
        instruction_duration_samples = 250 * 5
        instruction_samples_collected = 0

        while instruction_samples_collected < instruction_duration_samples:
            device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
            instruction_samples_collected += self.FrameLength
        excel_file_lable = pd.read_csv(f'Block{block_to_run}_key.csv')
            
        tdataarray=[]
        tdata=[]
        root.update()
        for j in range (0,40):
            row_data = excel_file_lable.iloc[j,[1, 2, 3]].to_numpy()
            self.image_window.next_image()
            for i in range(0, self.numberOfGetDataCalls): #self.numberOfGetDataCalls=250
                device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                key_pressed = self.key_detector.check_key_press()        
                dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength,
                combined_data = np.concatenate((data, row_data))
                combined_data = np.concatenate((combined_data,[key_pressed]))
                tdata.append(combined_data)
                tdataarray = np.array(tdata)    
            csv_filename = f'single_block_{block_to_run}.csv'
            if self.curr_phase.get() == "Pre-Evaluation":  
                csv_filepath = os.path.join(pre_folder, csv_filename)
            if self.curr_phase.get() == "Post-Evaluation":  
                csv_filepath = os.path.join(post_folder, csv_filename)
            with open(csv_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in tdataarray:
                    writer.writerow(row)
                    
        print(j)   
        del tdata
        del tdataarray
        self.image_window.pleaseWait_image()
        self.image_window.label.update_idletasks()  
        self.top.mainloop()
        self.curr_block.set(str(self.block))
        device.StopAcquisition() 
        
    def end_block(self):
        global image_window
        self.add_patient_data()
        self.image_window.close_window()
        device.StopAcquisition()
        print('Disconnected')
    
    def update_patient_data(self):
        self.add_patient_data()

if __name__ == "__main__":
    root = Tk()
    root.geometry("%dx%d+%d+%d" % (1000, 500, 100, 100))
    root.title("Root Window Controls")
    main_window = RootWindow(root)
    root.mainloop()
    



 
