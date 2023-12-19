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
        
        self.create_trial_but = Button(self.frame_2, text="Recalibrate Classifier", bg="yellow", font=10, command=self.run_RSVMScript)
        self.create_trial_but.grid(row=5, column=1, columnspan=2, pady=15, padx=5)
        
        self.end_trial_but = Button(self.frame_2, text="Score", bg="pink", font=("TkDefaultFont", 15), width=20, command=self.HBScore)
        self.end_trial_but.grid(row=6, column=0, columnspan=4, pady=20, padx=5)    

        self.end_trial_but = Button(self.frame_2, text="End Trial", bg="red", font=("TkDefaultFont", 15), width=20, command=self.end_trial)
        self.end_trial_but.grid(row=7, column=0, columnspan=4, pady=20, padx=5)
        
        self.frame_2.grid(padx=30, pady=50, row=0, column=1)

        self.frame_3 = tk.Frame(master)

         
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

    def open_image_win(self):
        global image_window, top
        top = Toplevel()
        monitors = get_monitors()
        if len(monitors) >= 2:
            print('Two monitors have been detected')
            monitor = monitors[1]  # This is the second monitor
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

    def butter_bandpass_filter_base(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

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
        return df_new

    def reject_artifacts(self, df, channel):
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
        # frequency_bands = {'delta': (0.5, 4),'theta': (4, 8),'alpha': (8, 14),'beta': (14, 30),'gamma': (30, 40),'ERP':(0.4,40)}
        fs=250
      
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)
        
        if self.curr_phase.get() == "Neurofeedback":  
            seq_list = [1,2,3,4,5,6,7,8]
            print('seq_list', seq_list)
            print('Trial',self.block)
            self.patient_progress[2]=self.block+1
            print('randomized_blocks:',seq_list[self.block])

            device.StartAcquisition(False)
        
            image_window.pleaseWait_image()
            pw1=[]
            for pw in range(0, 1):
                for p in range(0, 2*self.numberOfGetDataCalls): #self.numberOfGetDataCalls=250
                    device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)         
                    dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                    data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength
                    pw1.append(data.copy())
                    pw1dataarray=np.array(pw1)    
                pw1_totdata_array = pw1dataarray.reshape(-1, 17) 
                pw1_data=pw1_totdata_array[:, :8]
            pw_data_1=pw1_data.copy()
            
            pw2=[]
            for pw in range(0, 1):
                for p in range(0, self.numberOfGetDataCalls): 
                    device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                    dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                    data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength
                    pw2.append(data.copy())
                    pw2dataarray=np.array(pw2)    
                pw2_totdata_array = pw2dataarray.reshape(-1, 17) 
                pw2_data=pw2_totdata_array[:, :8]
            pw_data_2=pw2_data.copy()


    
            image_window.instructions_image_nf()
            instdata=[]
            for pw3 in range(0, 5):
                for p in range(0, self.numberOfGetDataCalls): 
                    device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                    dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                    data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength
                    instdata.append(data.copy())
                    instdataarray=np.array(instdata)    
                new_totdata_array = instdataarray.reshape(-1, 17) 
                instLast_data=new_totdata_array[:, :8]
            instruction_data=instLast_data.copy()
            top.update()
            
            
            print('self.block', self.block)
            Report_Number = self.block
            folder_name = patient_name

            # Use an f-string to construct the file path
            model_filename  = fr'C:\Users\tnlab\OneDrive\Documents\GitHub\AlphaFold\Neurofeedback-Based-BCI\best_mlp_{Report_Number}_{folder_name}.joblib'
            # model_filename = r'C:\Users\tnlab\OneDrive\Documents\GitHub\AlphaFold\Neurofeedback-Based-BCI\best_mlp_{Report_Number}_{folder_name}.joblib'
            loaded_model = joblib.load(model_filename)
            print('model_filename', model_filename)
            
            # Initialize the buffer
            num_columns_nf = 8
            buffer_size_seconds = 5
            samples_per_second = 250 
            buffer_size_samples = buffer_size_seconds * samples_per_second
            buffer = np.zeros((buffer_size_samples, 8))  # 8 is the number of EEG channels
            filter_states = [None] * num_columns_nf
            face_alpha_values = [0,70,128,204,255] 
            face_alpha_index=2
            current_directory = os.getcwd()
            # print(f"Current directory: {current_directory}")

            final_lable_array=[]
            raw=[]
            PP=[]
            base=[]
            for j in range (0,8):
                selected_columns = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'Po7', 'Oz', 'Po8']
                tdata=[]
                lable=[]
                filter_states = [None] * num_columns_nf
                print('j=', j)
                
                if j==0:
                    image_window.display_gray_image()
                    for n in range(0,7): 
                        print('n', n)
                        for i in range(self.numberOfGetDataCalls): 
                            device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                            dataa= np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                            data=np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength
                            tdata.append(data.copy())
                            
                            tdataarray=np.array(tdata)    
                        new_totdata_array= tdataarray.reshape(-1, 17) 
                        Last_data=new_totdata_array[:, :8]
                        raw.append(Last_data)
                      
                        label_array= np.zeros((250, 4), dtype=object) 
                        label_array.fill('G')
                        lable.append(label_array)
                        
                        nplable=np.array(lable).reshape(-1, 4)
                        fal = np.concatenate((new_totdata_array, nplable), axis=1)
                    
                    buffer = np.append(buffer, Last_data, axis=0)
                    if buffer.shape[0] > buffer_size_samples:
                        num_extra_samples = buffer.shape[0] - buffer_size_samples
                        buffer = buffer[num_extra_samples:, :]
                    else:
                        buffer = Last_data[-buffer_size_samples:, :]  
                    
                    del tdata
                    
                    final_lable_array.append(fal)
                    fl=np.array(final_lable_array).reshape(-1, 21)
                    
                    grey=Last_data.copy()
                    
                    base=np.vstack((pw_data_2, instruction_data, grey))
                    # print('base', base.shape)

                    base_bp=np.copy(base)
                    num_columns_nf = buffer.shape[1]
                    for column in range(num_columns_nf):
                        base_bp[:, column], filter_states[column] = self.butter_bandpass_filter(
                            base_bp[:, column], lowcut=.4, highcut=40, fs=250, order=5, initial_state=filter_states[column])
                    base_bp = pd.DataFrame(base_bp)

                    # 2. Artifact rejection
                    base_artifact_RJ = base_bp.copy()
                    for channel in range (8):
                        base_artifact_RJ= self.reject_artifacts(base_artifact_RJ, channel)     
                        
                    # 4. Denoising and other preprocessing
                    base_artifact_RJ.columns = selected_columns
                    base_df_denoised = self.preprocess(pd.DataFrame(base_artifact_RJ), col_names=selected_columns, n_clusters=[50]*len(selected_columns)) 
                    # print('base_df_denoised', base_df_denoised.shape) 
                    
                    base_split=base_df_denoised.iloc[-1750:,]
                    # print('base_split.shape', base_split.shape)
                    base_mean=np.mean(base_split, axis=0)
                    # print('base_mean', type(base_mean), base_mean.shape, base_mean)
                    
                    b_int=np.copy(buffer)
                    num_columns_nf = buffer.shape[1]
                    
         
                    # final_lable_array.append(fal)
                    # final_lable_array_np = np.concatenate(final_lable_array, axis=0) if final_lable_array else np.empty((0, 21))
                    # # print('final_lable_array_np.shape', final_lable_array_np.shape)
                    # fl=np.array(final_lable_array_np).reshape(-1, 21)

                else:
                    image_window.start_new_trial()
                    for n in range(0,5): #looking at each image for 5 seconds
                        print('n', n)
                        for i in range(self.numberOfGetDataCalls): 
                            device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                            dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                            data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength
                            tdata.append(data.copy())
                            tdataarray=np.array(tdata)
                                
                        new_totdata_array = tdataarray.reshape(-1, 17) 
                        Last_data=new_totdata_array[:, :8]
                        raw.append(Last_data)
                        


                        buffer = np.append(buffer, Last_data[-250:, :], axis=0)
                        if buffer.shape[0] > buffer_size_samples:
                            num_extra_samples = buffer.shape[0] - buffer_size_samples
                            buffer = buffer[num_extra_samples:, :]
                        df = pd.DataFrame(buffer)
                        Combined_raw_eeg_nf_bp = np.copy(buffer)
                        num_columns_nf = buffer.shape[1]
                        
                        for column in range(num_columns_nf):
                            Combined_raw_eeg_nf_bp[:, column], filter_states[column] = self.butter_bandpass_filter(
                                Combined_raw_eeg_nf_bp[:, column], lowcut=.4, highcut=40, fs=250, order=5, initial_state=filter_states[column])
                        combined_raw_eeg_nf_bp = pd.DataFrame(Combined_raw_eeg_nf_bp)

                        # 2. Artifact rejection
                        BP_artifact_RJ = combined_raw_eeg_nf_bp.copy()
                        initial_BP_artifact_RJ = BP_artifact_RJ.iloc[:-(n+1)*250]
                        for channel in range (8):
                            BP_artifact_RJ= self.reject_artifacts(BP_artifact_RJ.iloc[-(n+1)*250:], channel)     
                        DN=pd.concat([initial_BP_artifact_RJ, BP_artifact_RJ], axis=0)
                        
                        # 4. Denoising and other preprocessing
                        DN.columns = selected_columns
                        eeg_df_denoised = self.preprocess(DN, col_names=selected_columns, n_clusters=[50]*len(selected_columns))                            
                        # print('eeg_df_denoised', type(eeg_df_denoised), eeg_df_denoised.shape)
                        eeg_base_corrected=eeg_df_denoised.subtract(base_mean, axis=1)
                        # print('eeg_base_corrected', eeg_base_corrected.shape)

                        denoised=pd.DataFrame(eeg_df_denoised)
                        # print('denoised',denoised.shape)

                        chunks = np.array_split(eeg_base_corrected.to_numpy(), 5, axis=0)                    
                        eeg_signal = chunks[4].reshape(8, 250)  # reshaped to (8, 250)
                        Xn=eeg_signal.reshape(-1,2000)
                        predictions =loaded_model.predict(Xn)
                        instruction = self.instruction_mapping[seq_list[self.block]]
                        correct_prediction = (instruction == 'Face' and predictions[0] == 0) or (instruction == 'Scene' and predictions[0] == 1)
                        label_array = np.zeros((250, 4), dtype=object) 
                        label_array[:, 2] = 'F' if instruction == 'Face' else 'S'                    
                        for row in range(label_array.shape[0]):
                            if (instruction == 'Face' and correct_prediction):
                                label_array[row, 0] = 'F'
                            if (instruction == 'Scene' and correct_prediction):
                                label_array[row, 0] = 'S'
                            elif (instruction == 'Face' and not correct_prediction) or (instruction == 'Scene' and not correct_prediction):
                                label_array[row, 0] = 'N'
                        label_array[:, 1] =label_array[:, 0]         
                        label_array[:, 3] = 1 if correct_prediction else 0        
                        
                        # Adjust alpha
                        if instruction == 'Face':
                            if correct_prediction:
                                face_alpha_index = min(face_alpha_index + 1, len(face_alpha_values) - 1)
                            else:
                                face_alpha_index = max(face_alpha_index - 1, 0)
                        else:  # Instruction is 'Scene'
                            if correct_prediction:
                                face_alpha_index = max(face_alpha_index - 1, 0) 
                            else:
                                face_alpha_index = min(face_alpha_index + 1, len(face_alpha_values) - 1)
                        
                        new_face_alpha=face_alpha_values[face_alpha_index]
                        image_window.update_transparency(new_face_alpha)
                        
                        lable.append(label_array)
                        nplable=np.array(lable).reshape(-1, 4)
                        fal = np.concatenate((new_totdata_array, nplable), axis=1)
                        # print('nplable', nplable.shape)
                   

                    image_window.display_gray_image()
                    for n in range(0,3): #looking at each image for 5 seconds
                        print('n', n)
                        for i in range(self.numberOfGetDataCalls): 
                            device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                            dataa= np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                            data= np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength
                            tdata.append(data.copy())
                            tdataarray=np.array(tdata)
                                
                        new_totdata_array = tdataarray.reshape(-1, 17)
         
                        
                        label= np.zeros((250, 4), dtype=object) 
                        label.fill('r') 
                        lable.append(label)
                        nplable=np.array(lable).reshape(-1, 4)
                    
                    print('tdata', len(tdata), 'new_totdata_array.shape', 'j=', j,  new_totdata_array.shape)    

                    fal = np.concatenate((new_totdata_array, nplable), axis=1)
                    
                    # fal_f=np.vstack((fal, rest ))
                    
                    # del tdata
                    final_lable_array.append(fal)                   
                    final_lable_array_np = np.concatenate(final_lable_array, axis=0) if final_lable_array else np.empty((0, 21))
                    
                    # print('final_lable_array_np.shape', final_lable_array_np.shape)
                    fl=np.array(final_lable_array_np).reshape(-1, 21)
                    
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
            total_blocks = len(seq_list)
            device.StartAcquisition(False)
            # self.receiveBufferBufferLength = 25600000
            self.receiveBuffer = bytearray(self.receiveBufferBufferLength)
            
            tpw_0=[]
            image_window.pleaseWait_image()
            for pw in range(0, 15):
                for p in range(0, 2*self.numberOfGetDataCalls): #self.numberOfGetDataCalls=250
                    device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                    pw_0 = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                    pw_np_0 = np.reshape(pw_0, (self.numberOfAcquiredChannels,))  # Ensure correct tuple format for reshape
                    tpw_0.append(pw_np_0.tolist())  # Convert to list before appending
                tpw_np_0 = np.array(tpw_0)
                
                csv_filename = f'pw_0_.csv'
                csv_filepath = os.path.join(pre_folder if self.curr_phase.get() == "Pre-Evaluation" else post_folder, csv_filename)
                with open(csv_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in tpw_np_0:
                        writer.writerow(row)
            del tpw_0
            del tpw_np_0  
            
            tpw=[]
            for pw2 in range(0, 30):
                for p in range(0, self.numberOfGetDataCalls): #self.numberOfGetDataCalls=250
                    device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                    pw = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                    pw_np = np.reshape(pw, (self.numberOfAcquiredChannels,))  # Ensure correct tuple format for reshape
                    tpw.append(pw_np.tolist())  # Convert to list before appending                
                tpw_np = np.array(tpw)
                
                csv_filename = f'pw_2_.csv'
                csv_filepath = os.path.join(pre_folder if self.curr_phase.get() == "Pre-Evaluation" else post_folder, csv_filename)
                with open(csv_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in tpw_np:
                        writer.writerow(row)
            del tpw
            del tpw_np
                 
            for self.block in range(total_blocks):
                print('total_blocks:', total_blocks)
                if self.curr_phase.get() == 'Pre-Evaluation':
                    self.patient_progress[1]=self.block+1
                else:
                    self.patient_progress[3]=self.block+1    
                print('randomized_blocks:', seq_list[self.block])
                tdata_inst=[]
                image_window.instructions_image()
                top.update()

                for p in range(0, 5* self.numberOfGetDataCalls):  # self.numberOfGetDataCalls=250
                    device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                    dataa_inst = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                    data_inst = np.reshape(dataa_inst, (self.numberOfAcquiredChannels,))  # Ensure correct tuple format for reshape
                    tdata_inst.append(data_inst.tolist())  # Convert to list before appending
                
                data_inst_np = np.array(tdata_inst)
                print("Sample data:", data_inst_np[:5])

                csv_filename = f'data_inst_np_{self.block}_{seq_list[self.block]}.csv'
                csv_filepath = os.path.join(pre_folder if self.curr_phase.get() == "Pre-Evaluation" else post_folder, csv_filename)

                with open(csv_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in data_inst_np:
                        writer.writerow(row)
                del tdata_inst
                del data_inst_np

                excel_file_lable = pd.read_csv(f'Block{seq_list[self.block]}_key.csv')
                tdataarray=[]
                tdata=[]
                root.update()
                
                for j in range (0,41):
                    row_data = excel_file_lable.iloc[j,[1, 2, 3]].to_numpy()
                    print('row_data', row_data)
                    image_window.next_image()  
                    if j==0:
                        for i in range(0, 7*self.numberOfGetDataCalls): #self.numberOfGetDataCalls=250
                            device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                            key_pressed = self.key_detector.check_key_press()
                            # Convert receive buffer to numpy float array 
                            dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                            data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength,
                            combined_data = np.concatenate((data, row_data))
                            combined_data = np.concatenate((combined_data,[key_pressed]))
                            tdata.append(combined_data)
                            tdataarray = np.array(tdata)    

                        csv_filename = f'raw_eeg_block_{self.block}_{seq_list[self.block]}.csv'
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

                        csv_filename = f'raw_eeg_block_{self.block}_{seq_list[self.block]}.csv'
                        if self.curr_phase.get() == "Pre-Evaluation":  
                            csv_filepath = os.path.join(pre_folder, csv_filename)
                        if self.curr_phase.get() == "Post-Evaluation":  
                            csv_filepath = os.path.join(post_folder, csv_filename)
                        with open(csv_filepath, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for row in tdataarray:
                                writer.writerow(row)       
                    print('j:', j) 
                del tdata
                del tdataarray
                tdata_pw=[]
                image_window.pleaseWait_image()
                
                for k in range(0, 10*self.numberOfGetDataCalls): #self.numberOfGetDataCalls=250
                    device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                    dataa_pw = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                    data_pw = np.reshape(dataa_pw, (self.numberOfAcquiredChannels)) #self.FrameLength,
                    tdata_pw.append(data_pw.tolist())
                data_pw_np = np.array(tdata_pw)    

                csv_filename = f'data_pw_np_{self.block}_{seq_list[self.block]}.csv'
                if self.curr_phase.get() == "Pre-Evaluation":  
                    csv_filepath = os.path.join(pre_folder, csv_filename)
                if self.curr_phase.get() == "Post-Evaluation":  
                    csv_filepath = os.path.join(post_folder, csv_filename)
                with open(csv_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in data_pw_np:
                        writer.writerow(row)    
                del tdata_pw
                del data_pw_np
                self.update_gui()
                self.update_patient_data()
            
               
                # self.receiveBuffer = np.empty_like(self.receiveBuffer)
        image_window.pleaseWait_image()   
        self.update_gui()
        self.update_patient_data() 
        device.StopAcquisition()      

    ################################################################################################################################    
    ################################################################################################################################
    ################################################################################################################################  
    def Create_SVMScript(self):
        try:
            subprocess.run(["python", "MLP_1.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"The MLP_1.py script encountered an error: {e}")
        except FileNotFoundError:
            print("The MLP_1.py script was not found.")

    def run_RSVMScript(self):
        try:
            subprocess.run(["python", "MLP_1.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"The MLP_1.py script encountered an error: {e}")
        except FileNotFoundError:
            print("The MLP_1.py script was not found.")
            
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
    



 

