# LSTM_MVAE_Wavelet
Execution of LSTM_MVAE:
1.	Execute: python MVAE_Mamography.py

The script files are: MVAE_Mamography.py
		      MVAE_NSLKDD_attack_1_Mixing.py
		      MVAE_Pendigit.py
		      MVAE_Shuttle.py


Comparison with OmniAnomaly:


Generate the .txt dataset for Mammography:
1.	Execute the Data_Prepare.py under the folder OmniAnomaly_Mamography_Dataset
2.	It will generate 3 txt files: MamoTest-1-1.txt, MamoTestLabel-1-1.txt, MamoTrain-1-1.txt
3.	Move those files into test, test_label, train folders, respectively.
4.	Rename all these files in their respective folders as Mamo-1-1.txt


Generate the .pkl files from the .txt files for processing by the OmniAnomaly Python codes 
1.	Download and Setup environment for the OmniAnomaly codes from https://github.com/smallcowbaby/OmniAnomaly
2.	Copy and Paste the utility file : OmniAnomaly_Utils_Mammo.py into the utility folder of “omni_anomaly” 
3.	Execute “python OmniAnomaly_Data_Preprocess_Mamo.py mamo”
Outcome: the resultant .pkl will be generated into some user given folders.
Final execution of evaluation: “python OmniAnomaly_Main_Mamo.py” 

