import pandas as pd
import matplotlib.pyplot as plt
import glob, os
import scipy.io.wavfile
import scipy.signal
import scipy.stats as stats
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.utils import shuffle
import os
import subprocess
import platform

########################################################################################################################################################################
#Creating the data frame

# Input the folder name
folder_name = input("Please enter the subject name: ")

# Define the column names
column_names = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ',
                'Battery', 'Sample', 'FZ', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CPZ', 'PZ', 'Unknown', 'Instruction', 'Female/Male', 'Outdoor/Indoor', 'Human Behavior']

# Create an empty DataFrame 
df = []
# Check if the folder exists
if os.path.exists(folder_name) and os.path.isdir(folder_name):
    # Iterate through the files in the folder
    for file_name in os.listdir(folder_name):
        if file_name.endswith('.csv'):
            # Read the CSV file
            file_path = os.path.join(folder_name, file_name)
            print('file_path',file_path)
            df_temp = pd.read_csv(file_path)
            # print(df_temp)

            # Print the file name and the shape of the read data
            print(f"Processing {file_name}: {df_temp.shape}")

            # Append the data to the main DataFrame
            df.append(df_temp)
            
    combined_data_array_3d = np.array(df)
    combined_data_array_2d= combined_data_array_3d.reshape(8 * 9999, 21)
    # Print the final DataFrame shape
    print(f"Final DataFrame shape: {combined_data_array_2d.shape}")
    
# Convert the NumPy array to a pandas DataFrame
Combined_raw_eeg = pd.DataFrame(combined_data_array_2d)
Combined_raw_eeg.columns = column_names

#Excluding the useless columns
columns_to_remove = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'Battery', 'Sample', 'Unknown','Female/Male', 'Outdoor/Indoor', 'Human Behavior']
Combined_raw_eeg = Combined_raw_eeg.drop(columns=columns_to_remove, axis=1)
print(Combined_raw_eeg.shape)

# # Save the DataFrame to an Excel file
# output_file_name = "Combined_raw_eeg_excluded.xlsx"
# output_file_path = os.path.join(folder_name, output_file_name)
# Combined_raw_eeg.to_excel(output_file_path, index=False)
   
 
# # Open the folder using the default file explorer
# if platform.system() == "Windows":
#         subprocess.run(["explorer", folder_name])
# elif platform.system() == "Darwin":  # macOS
#         subprocess.run(["open", folder_name])
# elif platform.system() == "Linux":
#         subprocess.run(["xdg-open", folder_name])
# else:
#         print("Unsupported platform.")

####################################################################################################################################################################
#Event
# Create a new column 'event'
Combined_raw_eeg['event'] = ''

# Iterate over the rows and assign values to the 'event' column based on the 'Instruction' column
for index, row in Combined_raw_eeg.iterrows():
    if row['Instruction'] == 'F' or row['Instruction'] == 'M':
        Combined_raw_eeg.at[index, 'event'] = 'F'
    elif row['Instruction'] == 'I' or row['Instruction'] == 'O':
        Combined_raw_eeg.at[index, 'event'] = 'S'
print(Combined_raw_eeg.shape)

# # Save the DataFrame to an Excel file
output_file_name = "Combined_raw_eeg_excluded.xlsx"
output_file_path = os.path.join(folder_name, output_file_name)
Combined_raw_eeg.to_excel(output_file_path, index=False)

# Select useful columns
eeg_df = Combined_raw_eeg[["FZ", "FC1", "FC2", "C3", "CZ", "C4", "CPZ", "PZ"]]
print('eegdef' ,eeg_df )
#####################################################################################################################################################################
#Pre-proccessing

# Denoising 
def denoise_data(df, col_names, n_clusters):
    df_denoised = df.copy()
    for col_name, k in zip(col_names, n_clusters):
        # Convert column to numeric format
        df_denoised[col_name] = pd.to_numeric(df_denoised[col_name], errors='coerce')
        # Select only numeric columns
        X = df_denoised.select_dtypes(include=['float64', 'int64'])
        # Fit KNeighborsRegressor
        clf = KNeighborsRegressor(n_neighbors=k, weights='uniform')
        clf.fit(X.index.values[:, np.newaxis], X[col_name])
        # Predict values
        y_pred = clf.predict(X.index.values[:, np.newaxis])  
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


# This cell is used for applying the preprocessing functions and concatinating the dataframes 
eeg_df_denoised = preprocess(eeg_df, col_names=list(eeg_df.columns), n_clusters=[50]*len(eeg_df.columns))
print('denoised', eeg_df_denoised)
print(eeg_df_denoised.shape)

# Concatenate the denoised dataframe with the original dataframe, including the 'event' column
eeg_df_denoised_with_event = pd.concat([eeg_df_denoised, Combined_raw_eeg['event']], axis=1)

# Save the DataFrame to an Excel file
output_file_name = "Denoised_eeg_event_included.xlsx"
output_file_path = os.path.join(folder_name, output_file_name)
eeg_df_denoised_with_event.to_excel(output_file_path, index=False)

# Remove Nan rows
eeg_df_denoised_with_event = eeg_df_denoised_with_event.dropna(axis=0)
print(eeg_df_denoised_with_event.shape)



# Select first 10,000 samples
first_10000_samples = eeg_df_denoised.iloc[1000:1500]
first_1000_samples = eeg_df.iloc[1000:1500]
################################################################################################################################################
# Plot each column separately for the first 10,000 samples
for column in first_10000_samples.columns:
    plt.figure()  # Create a new figure for each column
    plt.plot(first_10000_samples.index, first_10000_samples[column])
    plt.xlabel('Sample Number (Row Number)')
    plt.ylabel('Data')
    # Set the y-axis scale
    min_value = first_10000_samples[column].min()
    max_value = first_10000_samples[column].max()
    plt.ylim(min_value - 0.0001, max_value + 0.0001)

    plt.title(f'EEG Data for Column: {column} (First 10,000 Samples)')
    plt.show()

for column in first_1000_samples.columns:
    plt.figure()  # Create a new figure for each column
    plt.plot(first_1000_samples.index, first_1000_samples[column])
    plt.xlabel('Sample Number (Row Number)')
    plt.ylabel('Data')
    min_value = first_10000_samples[column].min()
    max_value = first_10000_samples[column].max()
    plt.ylim(min_value - 0.0001, max_value + 0.0001)

    plt.title(f'noisyEEG Data for Column: {column} (First 10,000 Samples)')
    plt.show()
