Order of Running the Scripts:
	1- composite_images.py: This script is used to create blocks of composite images.
	2- image_display_unicorn.py: Defines a class for displaying the composite images created by composite_images.py.
	3- gui-unicorn.py: Runs the attention training platform and records the EEG signal.
	4- DEEP_PROJ.ipynb: Houses the developed classification models.

Folder Descriptions:
	S1, ...,  S4: Each of these folders includes 8 Excel files that contain the EEG signals of a subject recorded in Section 3, along with the plots of their raw, bandpassed, 
	 	and denoised data.
	CNN: Contains the CNN model prediction results on the untouched data for each subject, their confusion matrix heatmap, and accuracy and loss plots.
	MLP: Contains the MLP model prediction results on the untouched data for each subject, their confusion matrix heatmap, and accuracy and loss plots.
	SVM: Contains the SVM model prediction results of the untouched data for each subject, their confusion matrix heatmap, and an Excel file detailing the 
	            model's accuracy and precision. This folder also includes three subfolders:
		HILBERT: Contains the results of classification using the Hilbert feature as input.
		HILBERT PLUS DATA: Contains the results of classification using a combination of data and the Hilbert feature as input.
		NO FEATURES: Contains the results of classification using data without extracting any features as input.
	Images: Contains the images used in the study.

Excel Files:
	evaluation_results_CNN: Summarizes the evaluation results of the CNN model for different subjects.
	evaluation_results_MLP: Summarizes the evaluation results of the MLP model for different subjects.
	Block1_key, Block2_key, ..., Block3_key: Contain the labels for the images in the corresponding block.