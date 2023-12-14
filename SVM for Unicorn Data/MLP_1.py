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
import pywt
from scipy.stats import skew, kurtosis
from scipy.signal import spectrogram
from mne.viz import plot_topomap
from scipy.signal import welch
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import optuna
from sklearn.datasets import make_classification
from PIL import Image, ImageDraw, ImageFont
from joblib import dump
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi

####################################################################################
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
    return df_new

def df_to_raw(df, sfreq=250):
    info = mne.create_info(ch_names=list(df.columns), sfreq=sfreq, ch_types=['eeg'] * df.shape[1])
    raw = mne.io.RawArray(df.T.values * 1e-6, info)  # Converting values to Volts from microvolts for MNE
    return raw

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

#########################################################################################
current_directory = os.getcwd()
patient_data_folder = os.path.join(current_directory, "2-Patient Data")

print(f"Current directory: {current_directory}")
print(f"Patient data folder: {patient_data_folder}")

folder_name = input("Please enter the subject name: ")
Report_Number = input("Please enter the reprt number: ")
full_folder_path = os.path.join(patient_data_folder, folder_name)

# root_folder = "2-Patient Data"
sub_folders = ["Pre Evaluation", "Neurofeedback", "Post Evaluation"]
phase = int(input("Enter the phase (0, 1, 2): "))  # Or however you get the phase value
# Determine which sub-folders to use based on the phase
folders_to_use = []
if phase == 0:
    folders_to_use = [sub_folders[0]]  # Just "Pre Evaluation"
elif phase == 1:
    folders_to_use = sub_folders[:2]  # "Pre Evaluation" and "Neurofeedback"
elif phase == 2:
    folders_to_use = [sub_folders[2]]  # 
print('folders_to_use:', folders_to_use)
# Iterate over each folder to read the csv files
selected_columns = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'Po7', 'Oz', 'Po8']

################################################################################################
duration = 40 
raw=[]
event=[]
BP=[]
PP=[]
B=[]
Human_Behavior=[]
for folder in folders_to_use:
    full_folder_path_ = os.path.join(full_folder_path, folder)
    print('full_folder_path_', full_folder_path_)
    if os.path.exists(full_folder_path_) and os.path.isdir(full_folder_path_):
        print(f"Reading from: {full_folder_path_}")
        for file_name in os.listdir(full_folder_path_):
            if file_name.endswith('.csv') and (file_name.startswith('raw_eeg_block') or file_name.startswith('fl_')):
                file_path = os.path.join(full_folder_path_, file_name)
                s_temp = pd.read_csv(file_path, header=None)
                inst = s_temp.iloc[:, 17]
                df_temp = s_temp.iloc[:, :8]
                raw.append(df_temp)
                event.append(inst)
                HB=s_temp.iloc[1750:, 17:21]
                inst = s_temp.iloc[:, 17]
                Human_Behavior.append(HB)
                
                # 1. Band Pass
                raw_bp = np.copy(df_temp)
                for column in range(8):
                    raw_bp[:, column] = butter_bandpass_filter(raw_bp[:, column], lowcut=.4, highcut=40, fs=250) 
                # print(raw_bp.shape)
                
                # 2. Artifact rejection
                BP_artifact_RJ = np.copy(raw_bp)
                for channel in range (8):
                    BP_artifact_RJ= reject_artifacts(pd.DataFrame(BP_artifact_RJ), channel)
                
                # 4. Denoising and other preprocessing
                BP_artifact_RJ.columns = selected_columns
                eeg_df_denoised = preprocess(pd.DataFrame(BP_artifact_RJ), col_names=selected_columns, n_clusters=[10]*len(selected_columns))
                baseline=eeg_df_denoised.iloc[:1750,]
                dd=eeg_df_denoised.iloc[1750:,]
                print(dd.shape)
                # eeg_df_denoised.plot(subplots=True, figsize=(15, 10), title='Denoised EEG Data')
                # plt.show()
                B.append(baseline)
                PP.append(dd)
    else:
        print(f"{full_folder_path_} does not exist")

#######################################################################################################################################################

# Define the new list to store baseline corrected data
baseline_corrected = []
for baseline, dd in zip(B, PP):
    baseline_avg = baseline.mean()
    corrected = dd -baseline_avg
    baseline_corrected.append(corrected)
    
baseline_corrected_np=np.array(baseline_corrected)
print('baseline_corrected_np.shape',baseline_corrected_np.shape)

event_np=np.array(event)
print('event_np.shape',event_np.shape)
label_np=event_np[:,1750:]
print('label_np.shape',label_np.shape)

fs=250
B_N=int(len(baseline_corrected)) #Number of blocks
PP_NP=baseline_corrected_np #shape: (B_N, 10000, 8=Channel Numbers)

EVENTS=label_np.reshape(B_N*(baseline_corrected_np.shape[1]), 1)
print('EVENTS', EVENTS)
denoised=PP_NP.reshape(B_N*(baseline_corrected_np.shape[1]), 8) # seprate each blocks' signal 
pp_sig_event=np.concatenate((denoised,EVENTS), axis=1) 


event_column_index = pp_sig_event.shape[1] - 1

# Create a boolean mask where the event is not 'n'
mask = pp_sig_event[:, event_column_index] != 'N'

# Apply the mask to filter out rows with event 'n'
pp_sig_event_filtered = pp_sig_event[mask]
pp_sig_event_no_event_column = pp_sig_event_filtered[:, :-1]


labels=[] 
face = [] #lable=0
scene=[]#lable=1
base=[] # label=2
# Aassuming correctness for the human behavior
for i in range(len(pp_sig_event_filtered)): #len(pp_sig_event) = the whole sample points, (df_temp.shape[0]*B_N)
    if 'M' in pp_sig_event_filtered[i, 8] or 'F' in pp_sig_event_filtered[i, 8]:
        face.append(pp_sig_event_filtered[i])
        labels.append(0)
    if 'I' in pp_sig_event_filtered[i, 8] or 'O' in pp_sig_event_filtered[i, 8] or 'S' in pp_sig_event_filtered[i, 8]:
        scene.append(pp_sig_event_filtered[i]) 
        labels.append(1)        
face = np.array(face)
print('face.shape', face.shape)
scene = np.array(scene)
print('scene.shape', scene.shape)
labels=np.array(labels) 
print('label.shape', labels.shape, labels)
###############################################################################################################
Human_Behavior_np=np.array(Human_Behavior).reshape(B_N*(baseline_corrected_np.shape[1]), 4)
denoised_im_ins_HB = np.concatenate((denoised, Human_Behavior_np), axis=1)

SCORE = []
for row in denoised_im_ins_HB:
    condition1 = (row[-4] == row[-3]) or (row[-4] == row[-2])
    condition2 = row[-1] == 1
    condition3 = (row[-4] != row[-3]) and (row[-4] != row[-2])
    condition4 = row[-1] == 0
    if (condition1 and condition2) or (condition3 and condition4):
        SCORE.append([1])
    else:
        SCORE.append([0])

print('score length', len(SCORE))
#score
win_size = 250
S = []
for i in range(0, len(SCORE), win_size):
    S_data = SCORE[i:i+win_size]
    S.append(S_data)
# print('s lenght', len(S))
# print(S)
S_np = np.array(S)
print('S_np shape', S_np.shape)
result_list = []

# Iterate through the "images" (first dimension)
for i in range(S_np.shape[0]):
    # Check if all 250 samples are 0
    if np.all(S_np[i, :, 0] == 0):
        result_list.append(0)
    else:
        result_list.append(1)
# print(result_list)
mean_value = sum(result_list) / len(result_list)
print("Mean of result list:", mean_value)
percentage_of_ones = mean_value * 100
rounded_percentage_of_ones = round(percentage_of_ones)
n=str(rounded_percentage_of_ones)
print('n', n)
img=Image.new('RGB', (1000,1000), color=(73,109,137))
d=ImageDraw.Draw(img)
font_0=ImageFont.truetype("arial.ttf", 500)
font_1=ImageFont.truetype("arial.ttf", 150)
d.text((150,50), "Your Score", font=font_1, fill=(255,255,0))
d.text((250,250), n, font=font_0, fill=(255,255,0))
img_file_name = f"Score.png"
img_file_path = os.path.join(full_folder_path_, img_file_name) 
img.save(img_file_path, index=False)

################################################################################################################
label=labels.reshape(int(labels.shape[0]/fs), fs)
Y=np.squeeze(label[:,0])
print('Y.shape', Y.shape)
denoised_reshaped = pp_sig_event_no_event_column.reshape(int(pp_sig_event_no_event_column.shape[0]/250), 250, 8)
print('denoised_reshaped.shape',denoised_reshaped.shape)


mlp_data=denoised_reshaped.reshape(denoised_reshaped.shape[0], denoised_reshaped.shape[1]*denoised_reshaped.shape[2])
print('mlp_data.shape', mlp_data.shape)

af_mlp=mlp_data
Y_mlp=np.squeeze(label[:,0])
print(af_mlp.shape, Y_mlp.shape)
af_mlp, Y_mlp= shuffle(af_mlp, Y_mlp)
print(af_mlp.shape, Y_mlp.shape)
# Balance the dataset
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled_mlp, y_resampled_mlp = oversampler.fit_resample(af_mlp, Y_mlp)
X_resampled_mlp= X_resampled_mlp.astype(np.float32)
y_resampled_mlp = y_resampled_mlp.astype(np.int32)

#Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X_resampled_mlp,y_resampled_mlp, test_size=0.1, random_state=42)

#Split to train and validation
X_train_mlp, X_validation_mlp, y_train_mlp, y_validation_mlp = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

########################################################################################################################################
def objective(trial):
    # Layers and neurons
    n_layers = trial.suggest_int('n_layers', 1,3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_layer{i}', 16,  512))
    
    # Activation function
    activation = trial.suggest_categorical('activation', ['relu', 'logistic', 'tanh', 'identity'])
    
    # Learning rate
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4,  1e-1, log=True)
    
    max_iter = trial.suggest_int('max_iter', 50, 1000)

    model = MLPClassifier(hidden_layer_sizes=tuple(layers), 
                          activation=activation, 
                          learning_rate_init=learning_rate_init,
                          max_iter=max_iter ,  # to ensure convergence in most cases
                          random_state=42)

    model.fit(X_train_mlp, y_train_mlp)

    # Evaluate
    predictions = model.predict(X_validation_mlp)
    accuracy = accuracy_score(y_validation_mlp, predictions)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

###################################################################################################################
# Extract the best parameters
best_params = study.best_params
# Extract individual parameters
n_layers = best_params['n_layers']
layers = [best_params[f'n_units_layer{i}'] for i in range(n_layers)]
activation = best_params['activation']
learning_rate_init = best_params['learning_rate_init']
max_iter = best_params['max_iter']
# Create the model using the best parameters
best_model = MLPClassifier(hidden_layer_sizes=tuple(layers), 
                           activation=activation, 
                           learning_rate_init=learning_rate_init,
                           max_iter=max_iter ,  # to ensure convergence in most cases
                           random_state=42)
# Train the model using training data
best_model.fit(X_train, y_train)
# Predict using the test data
predictions = best_model.predict(X_test)
# Predict using the training data
train_predictions = best_model.predict(X_train)
# Evaluate the model using training data
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Training Accuracy: {train_accuracy:.4f}")
# Evaluate the model using test data
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.4f}")

####################################################################################################################
report_mlp = classification_report(y_test, predictions, output_dict=True)
report_df_mlp = pd.DataFrame(report_mlp).transpose()
report_df_mlp.loc['accuracy', :] = [accuracy, None, None, None]
report_file_name = f"Report_{Report_Number}.xlsx"  # This becomes "Report_001.xlsx"
full_file_path = os.path.join(full_folder_path_, report_file_name)  
report_df_mlp.to_excel(full_file_path, index=False)

######################################################################################################################
# Save the model to a file
dump(best_model, f"best_mlp_{Report_Number}_{folder_name}.joblib")
