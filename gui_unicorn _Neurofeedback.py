
from tkinter import ttk
from tkinter import *
import csv
import pandas as pd
import numpy as np
from image_display_unicorn_NF import *
import UnicornPy
import random
device = UnicornPy.Unicorn("UN-2021.05.37")
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
from scipy.signal import butter, filtfilt
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
from PIL import Image, ImageDraw, ImageFilter


class RootWindow:
    
    def __init__(self, master):
        self.SamplingRate = UnicornPy.SamplingRate
        self.SerialNumber = 'UN-2021.05.37'
        self.numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()
        self.configuration = device.GetConfiguration()
        self.AcquisitionDurationInSeconds = 1 
        self.FrameLength=1
        self.numberOfGetDataCalls = int(self.AcquisitionDurationInSeconds * self.SamplingRate / self.FrameLength)
        self.receiveBufferBufferLength = self.FrameLength * self.numberOfAcquiredChannels * 4  # 4 bytes per float32
        self.receiveBuffer = bytearray(self.receiveBufferBufferLength)
        self.eeg_data = []
        self.image_window = None
        self.block = 0
        self.image_window_open = False
        self.patient_progress = ['', '0', '0', '0', '00000000']
        self.patient_index = 0
        self.patient_data_list = []
        self.pre_eval = 0
        self.neuro = 0
        self.post_eval = 0
        self.seq = None
        self.top = None
        self.r = [1, 2, 3, 4, 5, 6, 7, 8]
        self.key_pressed = False
        self.instruction_mapping = {1: 'Face', 2: 'Scene', 3: 'Face', 4: 'Scene', 5: 'Face', 6: 'Scene', 7: 'Face', 8: 'Scene'}
        
        master.bind('<KeyPress>', self.key_press)
        with open('pat_progess_v2.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.patient_data_list.append(row)
        # print('patient_data_list:)',self.patient_data_list)
        df = pd.read_csv('pat_progess_v2.csv')
        # print('patient_data_frame:',df)
        self.create_gui(master)

    
    def key_press(self,event):
        self.key_pressed = True
        
    def create_gui(self, master):
        self.frame_1 = tk.Frame(master)

        self.patient_name_label = tk.Label(self.frame_1, text="Patient Name:", font=15)
        self.patient_name_label.grid(row=1, column=0, pady=15, padx=5)
        self.patient_name_data = tk.StringVar()
        self.patient_name_entry = tk.Entry(self.frame_1, width=30, font=15)
        self.patient_name_entry.grid(row=1, column=1, pady=15, padx=5)

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

        self.create_trial_but = Button(self.frame_2, text="Create Trial", bg="green", font=15, command=self.create_trial)
        self.create_trial_but.grid(row=1, column=0, columnspan=2, pady=15, padx=5)

        self.start_trial_but = Button(self.frame_2, text="Start Trial", bg="green", font=15, command=self.start_trial)
        self.start_trial_but.grid(row=4, column=0, pady=15, padx=5)

        self.end_trial_but = Button(self.frame_2, text="End Trial", bg="red", font=15, command=self.end_trial)
        self.end_trial_but.grid(row=4, column=1, pady=15, padx=5)

        self.frame_2.grid(padx=30, pady=50, row=0, column=1)

        self.frame_3 = tk.Frame(master)

        self.single_block = tk.Label(self.frame_3, text="Single Block", font=8)
        self.single_block.grid(row=2, column=0, pady=5, padx=3)
        # blist = ["1", "2", "3", "4", "5", "6", "7", "8"]
        blist = ["1", "2", "3", "4", "5", "6"]
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
        # Scale progress to match the range of the progress bar (0-100)
        scaled_progress = (progress / 6) * 100
        self.progress.set(scaled_progress)
    
               
    def create_trial(self):
        patient_name = self.patient_name_entry.get()
        self.patient_progress[0] = patient_name
        random.shuffle(self.r)
        randomized_blocks = self.r
        # print('randomized_blocks', randomized_blocks)
        for data_list in self.patient_data_list:
            if patient_name == data_list[0]:
                self.patient_index = self.patient_data_list.index(data_list)
                self.patient_progress = data_list
                self.pre_eval = int(self.patient_progress[1])
                self.neuro = int(self.patient_progress[2])
                self.post_eval = int(self.patient_progress[3])
                self.seq = self.patient_progress[4]
                # self.patient_data_list.append(self.patient_progress)
                # print(f"Data Already Exists: {self.patient_progress}")
                if self.curr_phase.get() == 'Pre-Evaluation':
                    self.block = self.pre_eval
                elif self.curr_phase.get() == "Neurofeedback":
                    self.block = self.neuro
                elif self.curr_phase.get() == "Post-Evaluation":
                    self.block = self.post_eval
                self.curr_block.set(str(self.block))
                self.progress.set(round(self.block * 12.5))
                self.open_image_win()
                self.start_trial_but.config(state="normal")
                return
        
        self.pre_eval, self.neuro, self.post_eval = 0, 0, 0
        self.seq = " ".join(str(x) for x in randomized_blocks)
        self.patient_progress[4] = self.seq
        self.patient_data_list.append(self.patient_progress)
        self.patient_index = len(self.patient_data_list) - 1
        self.add_patient_data()
        # print(f"New Data Added: {self.patient_progress}")
        self.open_image_win()
        # self.update_gui()

        
    def add_patient_data(self):
        with open('pat_progess_v2.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in self.patient_data_list:
                writer.writerow(row)
        # print(f"Full Data added to CSV: {self.patient_data_list}")

    def open_image_win(self):
        global image_window, top
        top = Toplevel()
        top.geometry("%dx%d+%d+%d" % (800, 600, 950, 200))
        top.title("Image Slideshow")
        # print(list(self.seq.replace(" ", "")))
        image_window = DisplayImage(top, self.block, list(self.seq.replace(" ", "")))
        image_window.single_block = False
        image_window.create_instruct_order()
        image_window.create_img_arr()
        image_window.pleaseWait_image()
        self.image_window_open = True
        # print("Image Window Opened")
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

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        # print(f'lowcut: {lowcut}, highcut: {highcut}, fs: {fs}, order: {order}')
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    # Pre-proccessing
    # Denoising 
    def denoise_data(self, df, col_names, n_clusters):
        df_denoised = df.copy()
        df_denoised.reset_index(drop=True, inplace=True)
        for col_name, k in zip(col_names, n_clusters):
            # print(f"Processing column {col_name}")
            df_denoised[col_name] = pd.to_numeric(df_denoised[col_name], errors='coerce') # Convert column to numeric format
            X = df_denoised.select_dtypes(include=['float64', 'int64']) # Select only numeric columns
            clf = KNeighborsRegressor(n_neighbors=k, weights='uniform') # Fit KNeighborsRegressor
            clf.fit(X.index.values[:, np.newaxis], X[col_name])
            y_pred = clf.predict(X.index.values[:, np.newaxis]) # Predict values 
            df_denoised[col_name] = y_pred
        return df_denoised

    # Z_scoring
    def z_score(self, df, col_names):
        df_standard = df.copy()
        for col in col_names:
            df_standard[col] = (df[col] - df[col].mean()) / df[col].std()
        return df_standard

    # Detrending
    def detrend(self, df, col_names):
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
        df_new = self.detrend(df_new, col_names)
        return df_new    
        
    def start_trial(self):
        global image_window, top
        seq_list = [int(x) for x in self.seq if x.isdigit()]
        print(seq_list)
        print('Block',self.block)
        print('randomized_blocks:',seq_list[self.block])
        image_window.instructions_image()
        top.update()
        time.sleep(5)
        device.StartAcquisition(False)
        # instruction = self.instruction_mapping[seq_list[self.block]]
       
        # Load the trained SVM model
        filename = r'C:\Users\tnlab\OneDrive\Documents\GitHub\Neurofeedback-Based-BCI\my_svm_model.joblib'
        svm_model = joblib.load(filename)
        
        # Initialize the buffer
        buffer_size_seconds = 5
        samples_per_second = 250  # Assumed based on your description
        buffer_size_samples = buffer_size_seconds * samples_per_second
        buffer = np.zeros((buffer_size_samples, 8))  # 8 is the number of EEG channels
        face_alpha_values = [0,70,128,204,255] 
        face_alpha_index=2
        # face_alpha = 128
         
        for j in range (0,2):
            image_window.start_new_trial()

            for n in range(0,5): 
                tdata=[]
                for i in range(self.numberOfGetDataCalls): #looking at each image for 5 seconds
                    device.GetData(self.FrameLength, self.receiveBuffer, self.receiveBufferBufferLength)
                    dataa = np.frombuffer(self.receiveBuffer, dtype=np.float32, count=self.numberOfAcquiredChannels * self.FrameLength)
                    data = np.reshape(dataa, (self.numberOfAcquiredChannels)) #self.FrameLength
                    tdata.append(data)
                    tdataarray=np.array(tdata)
                new_totdata_array = tdataarray.reshape(-1, 17)  # Reshape the array into 2D
                Last_data=new_totdata_array[:, :8]
                buffer = np.append(buffer, Last_data, axis=0)
                if buffer.shape[0] > buffer_size_samples:
                    num_extra_samples = buffer.shape[0] - buffer_size_samples
                    buffer = buffer[num_extra_samples:, :]
                else:
                    buffer = Last_data[-buffer_size_samples:, :]        
                print('j',j, 'n',n)                     
                Combined_raw_eeg_nf_bp = np.copy(buffer)
                num_columns_nf = buffer.shape[1]
                for column in range(num_columns_nf):
                    Combined_raw_eeg_nf_bp[:, column] = self.butter_bandpass_filter(Combined_raw_eeg_nf_bp[:, column], lowcut=.4, highcut=40, fs=250, order=5)    
                combined_raw_eeg_nf_bp=pd.DataFrame(Combined_raw_eeg_nf_bp)
                eeg_df_denoised_nf = self.preprocess(combined_raw_eeg_nf_bp, col_names=list(combined_raw_eeg_nf_bp.columns), n_clusters=[50]*len(combined_raw_eeg_nf_bp.columns))
                denoised_data = eeg_df_denoised_nf.to_numpy()
                chunks = np.array_split(denoised_data, 5, axis=0)
                feature=[]
                scaler = StandardScaler()
                analytic_signal = hilbert(chunks[4])
                envelope = np.abs(analytic_signal)
                envelope=np.hstack((envelope, chunks[4]))
                envelop_standardized = scaler.fit_transform(envelope)
                envelop_standardized_tr=envelop_standardized.transpose()
                pca = PCA(n_components=16)  # how many components you want to keep
                pca.fit(envelop_standardized_tr)
                eeg_data_pca = pca.transform(envelop_standardized_tr)
                feature.append(eeg_data_pca)
                feature_array=np.array(feature)
                X_n=feature_array.reshape(-1,16*16)
                predictions =svm_model.predict(X_n)
                print('predictions', predictions)
                # Check if the prediction was correct
                instruction = self.instruction_mapping[seq_list[self.block]]
                correct_prediction = (instruction == 'Face' and predictions[0] == 0) or (instruction == 'Scene' and predictions[0] == 1)
                
                # Adjust alpha
                if instruction == 'Face':
                    if correct_prediction:
                        face_alpha_index = min(face_alpha_index + 1, len(face_alpha_values) - 1)
                        print(f"Face mask increased in the {n}th second.", 'face alpha index=', face_alpha_index)
                    else:
                        face_alpha_index = max(face_alpha_index - 1, 0)
                        print(f"Face mask decreased in the {n}th second.", 'face alpha index=', face_alpha_index)
                else:  # Instruction is 'Scene'
                    if correct_prediction:
                        face_alpha_index = max(face_alpha_index - 1, 0) 
                        print(f"Face mask decreased in the {n}th second.", 'face alpha index=', face_alpha_index)
                    else:
                        face_alpha_index = min(face_alpha_index + 1, len(face_alpha_values) - 1)
                        print(f"Face mask increased in the {n}th second.", 'face alpha index=', face_alpha_index)
                
                new_face_alpha=face_alpha_values[face_alpha_index]
                # image_window.create_composite_img(new_face_alpha) 
                image_window.update_transparency(new_face_alpha)

                del tdata
                print('j',j)    
                    
        image_window.pleaseWait_image()        
        self.update_gui()
        self.update_patient_data() 
        device.StopAcquisition() 
   
    ################################################################################################################################    
    ################################################################################################################################
    ################################################################################################################################  
    
    def end_trial(self):
        global image_window
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
        # config block or count then run next image function
        block_to_run = self.single_block_num_var.get()
        if self.image_window_open:
            image_window.close_window()

        self.top = Toplevel()
        self.top.geometry("%dx%d+%d+%d" % (800, 600, 950, 200))
        self.top.title("Image Slideshow")
        self.image_window = DisplayImage(self.top, self.block, block_to_run)
        self.image_window.single_block = True
        self.image_window.create_img_arr()
        self.image_window.pause = False
        self.image_window.next_image()
        self.image_window_open = True
        print("Image Window Opened")
        self.top.mainloop()

        self.curr_block.set(str(self.block))

    def end_block(self):
        self.image_window.close_window()
    
    def update_patient_data(self):
        self.add_patient_data()

if __name__ == "__main__":
    root = Tk()
    root.geometry("%dx%d+%d+%d" % (1000, 500, 100, 100))
    root.title("Root Window Controls")
    main_window = RootWindow(root)
    root.mainloop()
