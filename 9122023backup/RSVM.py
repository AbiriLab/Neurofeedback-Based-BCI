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

from sklearn import svm
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

#############################################################################
current_directory = os.getcwd()
patient_data_folder = os.path.join(current_directory, "2-Patient Data")

print(f"Current directory: {current_directory}")
print(f"Patient data folder: {patient_data_folder}")

folder_name = input("Please enter the subject name: ")
Report_Number = input("Please enter the reprt number: ")
# Phase = input("Please enter the phase:")

full_folder_path = os.path.join(patient_data_folder, folder_name)

##########################################################
column_names = ['FZ', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CPZ', 'PZ', 'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ',
                'Battery', 'Sample', 'Unknown', 'Instruction', 'Female/Male', 'Outdoor/Indoor', 'Human Behavior']
df = []

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
for folder in folders_to_use:
    full_folder_path_ = os.path.join(full_folder_path, folder)
    print(full_folder_path_)

    if os.path.exists(full_folder_path_) and os.path.isdir(full_folder_path_):
        print(f"Reading from: {full_folder_path_}")
        for file_name in os.listdir(full_folder_path_):
            if file_name.endswith('.csv'):
                file_path = os.path.join(full_folder_path_, file_name)
                df_temp = pd.read_csv(file_path, header=None)
                df.append(df_temp)
        combined_data_array_3d = np.array(df)
        print('combined_data_array_3d.shape', combined_data_array_3d.shape)
        combined_data_array_2d= combined_data_array_3d.reshape(-1, 21)
    else:
        print(f"{full_folder_path_} does not exist")

# # Process the collected data
# if df:
#     combined_data_array_3d = np.array(df)
#     combined_data_array_2d = combined_data_array_3d.reshape(-1, 21)
Combined_raw_eeg = pd.DataFrame(combined_data_array_2d)
Combined_raw_eeg.columns = column_names
columns_to_remove = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'Battery', 'Sample', 'Unknown','Instruction','Female/Male', 'Outdoor/Indoor', 'Human Behavior']

Combined_raw_eeg = Combined_raw_eeg.drop(columns=columns_to_remove, axis=1)
# else:
#     print("No data found.")

print( 'Combined_raw_eeg',  len(Combined_raw_eeg), type(Combined_raw_eeg))

##############################################################################

# Band pass filter
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

Combined_raw_eeg_bp = np.copy(Combined_raw_eeg)
num_columns = Combined_raw_eeg_bp.shape[1]
print(num_columns)
for column in range(num_columns):
    Combined_raw_eeg_bp[:, column] = butter_bandpass_filter(Combined_raw_eeg_bp[:, column], lowcut=.4, highcut=40, fs=250)    
#############################################################################################################################

# Pre-proccessing
# Denoising 
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

# Z_scoring
def z_score(df, col_names):
    df_standard = df.copy()
    for col in col_names:
        df_standard[col] = (df[col] - df[col].mean()) / df[col].std()
    return df_standard

# Detrending
def detrend(df, col_names):
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
    df_new = detrend(df_new, col_names)
    return df_new

Combined_raw_eeg_bp=pd.DataFrame(Combined_raw_eeg_bp)
eeg_df_denoised = preprocess(Combined_raw_eeg_bp, col_names=list(Combined_raw_eeg_bp.columns), n_clusters=[50]*len(Combined_raw_eeg_bp.columns))
##################################################################################################################################################
# Lableing
column_indices = {'Instruction': 17, 'Female/Male': 18, 'Outdoor/Indoor': 19, 'Human Behavior':20 }
selected_columns_HB = [column_indices['Instruction'], column_indices['Female/Male'], column_indices['Outdoor/Indoor'], column_indices['Human Behavior']]
selected_columns = [column_indices['Instruction'], column_indices['Female/Male'], column_indices['Outdoor/Indoor']]

data_im_ins = combined_data_array_2d[:, selected_columns]
data_im_ins_HB=combined_data_array_2d[:, selected_columns_HB]
denoised_im_ins = np.concatenate((eeg_df_denoised, data_im_ins), axis=1)
denoised_im_ins_HB = np.concatenate((eeg_df_denoised, data_im_ins_HB), axis=1)

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
print(result_list)

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


# img.save('report_file_name.png')
img.save(img_file_path, index=False)

# Check the third last column (column 9) and keep rows if column 9 is equal to 1
filtered_denoised_im_ins = denoised_im_ins[(denoised_im_ins[:, -3] == denoised_im_ins[:, -2]) | (denoised_im_ins[:, -3] == denoised_im_ins[:, -1])]
filtered_denoised_im_ins_df = pd.DataFrame(filtered_denoised_im_ins)



# Create a new column 'event'
filtered_denoised_im_ins_df['event'] = ''
for index, row in filtered_denoised_im_ins_df.iterrows():
    if row.iloc[-4] == 'F' or row.iloc[-4] == 'M':
        filtered_denoised_im_ins_df.at[index, 'event'] = '0'
    else: 
        filtered_denoised_im_ins_df.at[index, 'event'] = '1'
        
selected_data = filtered_denoised_im_ins_df.iloc[:, :8]  
lable=filtered_denoised_im_ins_df.iloc[:, -1:]
##################################################

win_size = 250
X = []
y = []

for i in range(0, len(selected_data), win_size):
    window_data = selected_data.iloc[i:i+win_size]
    window_label = lable.iloc[i:i+win_size]
    X.append(window_data)
    y.append(window_label)

X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y)
####################

array_3d = X.reshape(X.shape[0], 250*8)
# print(array_3d.shape)
print('X.shape', X.shape)
##############

# Hilbert feature extraction and PCA data Reduction
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()
feature=[]
for chunk in X:
    analytic_signal = hilbert(chunk)
    envelope = np.abs(analytic_signal)
    envelope=np.hstack((envelope, chunk))
    envelop_standardized = scaler.fit_transform(envelope)
    envelop_standardized_tr=envelop_standardized.transpose()
    pca = PCA(n_components=16)  # how many components you want to keep
    pca.fit(envelop_standardized_tr)
    eeg_data_pca = pca.transform(envelop_standardized_tr)
    # print(eeg_data_pca.shape)
    feature.append(eeg_data_pca)
print(len(feature))
feature_array=np.array(feature)
X_n=feature_array.reshape(-1,16*16)
# print(X_n.shape)
################

y_n=np.squeeze(y[:,0])
# print(X_n.shape, y_n.shape)

# Balance the dataset
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_n, y_n)
X_resampled= X_resampled.astype(np.float32)
y_resampled = y_resampled.astype(np.int32)

# Split X and y into training and testing sets
X_touched, X_untouch, y_touch, y_untouch = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_touched, y_touch, test_size=0.2, random_state=42)

# Convert y_train and y_test to categorical format for Keras
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
y_untouch=tf.keras.utils.to_categorical(y_untouch, num_classes=2)

# Convert one-hot-encoded labels to integer-encoded labels
y_train = np.argmax(y_train, axis=-1)
y_test = np.argmax(y_test, axis=-1)
y_untouch = np.argmax(y_untouch, axis=-1)

# print(y_train.shape, y_test.shape)

# print('X_train:', X_train.shape, 'y_train:', y_train.shape, 'X_test:', X_test.shape, 'y_test:',
#       y_test.shape, 'X_untouch:', X_untouch.shape, 'y_untouch:', y_untouch.shape )
####################################################################################

# Create a linear SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
filename = 'C:/Users/tnlab/OneDrive/Documents/GitHub/AlphaFold/Neurofeedback-Based-BCI/my_svm_model.joblib'
dump(clf, filename)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy: ', accuracy_score(y_test, y_pred))
report_svm_matrix = classification_report(y_test, y_pred)
print("Classification Report:")
print(report_svm_matrix)
report_svm = classification_report(y_test, y_pred, output_dict=True)
report_df_svm = pd.DataFrame(report_svm).transpose()
report_df_svm.loc['accuracy', :] = [accuracy, None, None, None]
report_file_name = f"Report_{Report_Number}.xlsx"  # This becomes "Report_001.xlsx"
full_file_path = os.path.join(full_folder_path_, report_file_name)  
report_df_svm.to_excel(full_file_path, index=False)