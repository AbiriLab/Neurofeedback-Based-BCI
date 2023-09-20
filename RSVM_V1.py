import os
import sklearn
import numpy as np
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

#############################################################################
current_directory = os.getcwd()
patient_data_folder = os.path.join(current_directory, "2-Patient Data")
print(f"Current directory: {current_directory}")
print(f"Patient data folder: {patient_data_folder}")
folder_name = input("Please enter the subject name: ")
Report_Number = input("Please enter the reprt number: ")
Phase = input("Please enter the phase:")
full_folder_path = os.path.join(patient_data_folder, folder_name)

##########################################################
N180_window = (5,10)
P300_window = (10,20) 
N500_window = (20, 30)
N600_window = (25, 35)
P650_window = (30,38)
P900_window = (40,50)
frequency_bands = {'delta': (0.5, 4),'theta': (4, 8),'alpha': (8, 14),'beta': (14, 30),'gamma': (30, 40),'ERP':(0.4,40) }

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def denoise_data(df, col_names, n_clusters):
    df_denoised = df.copy()
    for col_name, k in zip(col_names, n_clusters):
        df_denoised[col_name] = pd.to_numeric(df_denoised[col_name], errors='coerce') # Convert column to numeric format
        X = df_denoised.select_dtypes(include=['float64', 'int64']) # Select only numeric columns
        clf = KNeighborsRegressor(n_neighbors=k, weights='uniform') # Fit KNeighborsRegressor
        clf.fit(X.index.values[:, np.newaxis], X[col_name])
        y_pred = clf.predict(X.index.values[:, np.newaxis]) # Predict values 
        df_denoised[col_name] = y_pred
    return df_denoised

def z_score(df, col_names):
    df_standard = df.copy()
    for col in col_names:
        df_standard[col] = (df[col] - df[col].mean()) / df[col].std()
    return df_standard

def custom_detrend(df, col_names):
    df_detrended = df.copy()
    for col in col_names:
        y = df_detrended[col]
        x = np.arange(len(y))
        p = np.polyfit(x, y, 1)
        trend = np.polyval(p, x)
        detrended = y - trend
        df_detrended[col] = detrended
    return df_detrended

def preprocess(df, col_names, n_clusters):
    df_new = df.copy()
    df_new = denoise_data(df, col_names, n_clusters)
    df_new = z_score(df_new, col_names)
    df_new = custom_detrend(df_new, col_names)
    return df_new

def df_to_raw(df, sfreq=250):
    info = mne.create_info(ch_names=list(df.columns), sfreq=sfreq, ch_types=['eeg'] * df.shape[1])
    raw = mne.io.RawArray(df.T.values * 1e-6, info)  # Converting values to Volts from microvolts for MNE
    return raw

def reject_artifacts(df, channel):
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

def reject_artifacts_DN(df, channel):
    threshold_factor =5
    median = df[channel].median()
    mad = np.median(np.abs(df[channel] - median))
    spikes = np.abs(df[channel] - median) > threshold_factor * mad
    x = np.arange(len(df[channel]))
    cs = CubicSpline(x[~spikes], df[channel][~spikes])    # Interpolate using Cubic Spline
    interpolated_values = cs(x)
    interpolated_values[spikes] *= 0.5  # Make interpolated values 0.001 times smaller
    df[channel] = interpolated_values
    return df

def extract_ERP_features(epoch):
    N180_region= epoch[N180_window[0]:N180_window[1]]
    P300_region = epoch[P300_window[0]:P300_window[1]]
    N500_region = epoch[N500_window[0]:N500_window[1]]
    N600_region = epoch[N600_window[0]:N600_window[1]]
    P650_region = epoch[P650_window[0]:P650_window[1]]
    P900_region = epoch[P900_window[0]:P900_window[1]]
    N180_mean_amplitude = np.mean(N180_region)
    P300_mean_amplitude = np.mean(P300_region)
    N500_mean_amplitude = np.mean(N500_region)
    N600_mean_amplitude = np.mean(N600_region)
    P650_mean_amplitude = np.mean(P650_region)
    P900_mean_amplitude = np.mean(P900_region)
    return [
        N180_mean_amplitude,P300_mean_amplitude,N500_mean_amplitude,N600_mean_amplitude,P650_mean_amplitude,P900_mean_amplitude]

def apply_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def extract_ERP_for_band(signal, band_limits, fs=250):
    band_filtered_signal = apply_bandpass_filter(signal, band_limits[0], band_limits[1], fs)
    return extract_ERP_features(band_filtered_signal)

def extract_all_band_ERPs(signal, frequency_bands):
    all_band_features = {}
    for band_name, band_limits in frequency_bands.items():
        all_band_features[band_name] = extract_ERP_for_band(signal, band_limits)
    return all_band_features

def extract_all_band_ERPs_to_array(signal, frequency_bands):
    all_band_features_list = []
    for band_name, band_limits in frequency_bands.items():
        erp_features_for_band = extract_ERP_for_band(signal, band_limits)
        all_band_features_list.extend(erp_features_for_band)
    return np.array(all_band_features_list)

def calculate_hilbert_features(signal, fs):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = angle(analytic_signal)
    unwrapped_phase = unwrap(instantaneous_phase)
    instantaneous_frequency = diff(unwrapped_phase) / (2.0*np.pi) * fs 
    return amplitude_envelope, instantaneous_phase, instantaneous_frequency

def extract_hilbert_features(dataset, fs):
    n_channels = 8
    sampling_rate = 250
    all_hilbert_features = []
    for sample in dataset:
        channel_features = []
        for ch in range(n_channels):
            signal = sample[ch * sampling_rate: (ch + 1) * sampling_rate]
            amplitude_envelope, instantaneous_phase, instantaneous_frequency = calculate_hilbert_features(signal, fs)
            channel_features.append([
                np.mean(amplitude_envelope),
                # np.mean(instantaneous_phase),
                # np.mean(instantaneous_frequency)
            ])
        all_hilbert_features.append(channel_features)
    return np.array(all_hilbert_features)

###########################################################################################
selected_columns = ['Fz', 'FC1', 'FC2', 'C3', 'Cz', 'C4', 'CPz', 'Pz']
duration = 40 
raw=[]
event=[]
BP=[]
PP=[]
if os.path.exists(folder_name) and os.path.isdir(folder_name):
    for file_name in os.listdir(folder_name):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_name, file_name)
            s_temp = pd.read_csv(file_path, header=None)
            inst = s_temp.iloc[:, 17]
            df_temp = s_temp.iloc[:, :8]
            df_temp.columns = selected_columns 
            raw.append(df_temp)
            event.append(inst)
            raw_mne = df_to_raw(df_temp)
            # 1. Band Pass
            raw_mne_BP=raw_mne.filter(.4, 40)
            # 2. Artifact rejection
            BP_artifact_RJ = raw_mne_BP.copy()
            for channel in selected_columns:
                BP_artifact_RJ._data[selected_columns.index(channel), :] = reject_artifacts(pd.DataFrame(raw_mne._data.T, columns=selected_columns), channel)[channel].values
            # 3. Smoothing
            BP_artifact_RJ_SM=BP_artifact_RJ.copy()
            window_size = 10 
            for channel in selected_columns:
                channel_data = pd.Series(BP_artifact_RJ_SM._data[selected_columns.index(channel), :])
                BP_artifact_RJ_SM._data[selected_columns.index(channel), :] = channel_data.rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            BP.append(BP_artifact_RJ_SM)
            # 4. Denoising and other preprocessing
            eeg_df_denoised = preprocess(pd.DataFrame(BP_artifact_RJ_SM.get_data().T, columns=selected_columns), col_names=selected_columns, n_clusters=[50]*len(selected_columns))
            # 5. Artifact rejection on denoised data
            eeg_df_denoised_artifact_RJ = eeg_df_denoised.copy()
            for channel in selected_columns:
                eeg_df_denoised_artifact_RJ[channel] = reject_artifacts_DN(eeg_df_denoised, channel)[channel]     
            # 6. Smoothing the artifact-rejected denoised data
            window_size = 10  
            DN_SM = eeg_df_denoised_artifact_RJ.copy()
            for channel in selected_columns:
                channel_data = DN_SM[channel]
                DN_SM[channel] = channel_data.rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill'))
            PP.append(DN_SM)
            
###############################################################################################################
fs=250
B_N=int(len(PP))
PP_NP=np.array(PP)
event=np.array(event).reshape(B_N*(df_temp.shape[0]), 1)
denoised=PP_NP.reshape(B_N*(df_temp.shape[0]), 8)  
pp_sig_event=np.concatenate((denoised, event), axis=1) 
 
#############################################################
labels=[] 
face = [] #lable=0
scene=[]#lable=1
for i in range(len(pp_sig_event)):
    if 'M' in pp_sig_event[i, 8] or 'F' in pp_sig_event[i, 8]:
        face.append(pp_sig_event[i])
        labels.append(0)
    else:
        scene.append(pp_sig_event[i]) 
        labels.append(1)        
face = np.array(face)
scene = np.array(scene)
labels=np.array(labels)
                  
#############################################################################
F_N=int(B_N/2)
S_N=int(B_N/2)
face_eeg_dada=face.reshape(F_N,int(df_temp.shape[0]/fs),50,5,9)
face_eeg_dada=face_eeg_dada[:,:,:,:,:8]
face_eeg_dada=face_eeg_dada.mean(axis=3)
face_eeg_dada=face_eeg_dada.mean(axis=1)
face_mean=face_eeg_dada.mean(axis=0)
scene_eeg_dada=scene.reshape(S_N,int(df_temp.shape[0]/fs),50,5,9)
scene_eeg_dada=scene_eeg_dada[:,:,:,:,:8]
scene_eeg_dada=scene_eeg_dada.mean(axis=3)
scene_eeg_dada=scene_eeg_dada.mean(axis=1)
scene_mean=scene_eeg_dada.mean(axis=0)

############################################################################
X=denoised.reshape(int(denoised.shape[0]/fs), fs*8)
label=labels.reshape(int(labels.shape[0]/fs), fs)
Y=np.squeeze(label[:,0])
data = X

#############################################################################
hilbert_Fe=[]
for band, (low, high) in frequency_bands.items():
    filtered_signal = apply_bandpass_filter(data, low, high, fs)
    hilbert_bp=extract_hilbert_features(filtered_signal,fs)
    hilbert_Fe.append(hilbert_bp) 
hilbert_Fe_NP=np.array(hilbert_Fe)
hilbert_Fe_NP_transposed = hilbert_Fe_NP.transpose(1, 0, 2, 3)  # shape becomes (1600, 5, 8, 3)
Hilbert_FE= hilbert_Fe_NP_transposed.reshape(int(denoised.shape[0]/fs), 8, -1)

##################################################################################
data_reshaped = X.reshape(int(denoised.shape[0]/fs), 8, 250)
ERP_FE = np.array([[extract_ERP_features(data_reshaped[i, j, :]) for j in range(8)] for i in range(int(denoised.shape[0]/fs))])

###################################################################################
combined_features = np.concatenate([Hilbert_FE,ERP_FE], axis=2)
af=combined_features.reshape(int(denoised.shape[0]/fs), 8*combined_features.shape[2])
af, Y = shuffle(af, Y)

#####################################################################################
# Balance the dataset
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(af, Y)
X_resampled= X_resampled.astype(np.float32)
y_resampled = y_resampled.astype(np.int32)
X_touched, X_untouch, y_touch, y_untouch = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_touched, y_touch, test_size=0.1, random_state=42)
# Convert y_train and y_test to categorical format for Keras
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
y_untouch=tf.keras.utils.to_categorical(y_untouch, num_classes=2)
# Convert the data to a numerical type (float)
X_train = X_train.astype(np.float64)
# Convert one-hot-encoded labels to integer-encoded labels
y_train = np.argmax(y_train, axis=-1)
y_test = np.argmax(y_test, axis=-1)
y_untouch = np.argmax(y_untouch, axis=-1)

####################################################################################
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
# Save the model to disk
filename = 'C:/Users/tnlab/OneDrive/Documents/GitHub/Neurofeedback-Based-BCI/SVM for Unicorn Data/my_svm_model.joblib'
dump(clf, filename)

# Make predictions on the test set
y_pred = clf.predict(X_test)
print('Model accuracy: ', accuracy_score(y_test, y_pred))
report_svm_matrix = classification_report(y_test, y_pred)
print("Classification Report:")
print(report_svm_matrix)
report_svm = classification_report(y_test, y_pred, output_dict=True)

report_df_svm = pd.DataFrame(report_svm).transpose()
report_df_svm.to_excel(f"svm_classification_report_{folder_name}.xlsx", index=True)