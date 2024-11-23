# Here we have provided the complete instructions to apply the code for analysis of biomarker concentrations obtained from blood tests of different patients(Normal and cancer patients) by applying the methodology mentioned in our paper named "Enhanced Early Cancer Detection via Topological Data Analysis of Blood Biomarkers: A Non-Invasive Approach for Accurate, Multi-Cancer Screening" by Sudarshan Gogoi, Bikash Thakuri and Amit Chakraborty
# There are total 3 scripts written in Python: 1st script (file name: Script1.ipynb) contains characterization steps of different data types(Normal and different cancer types) based on the whole patients' data, 2nd script (file name:Script2.ipynb) contains characterization and detection steps of cancer at individual patient level plus detection of the cancer origin if it is stomach cancer, and 3rd script (file name:Script3.ipynb) contains detection of the cancer origin if it is pancreas cancer. By using 2nd and 3rd scripts, the analysis(cancer detection) is carried out for all the patients(Training and validation) together to save time.
# The codes are being executed in "jupyter" Notebook version 7.0.8 within Anaconda Navigator. All steps, described in the Methodology section of the paper, are briefed below:

# Script1.ipynb(Data file used: "clinical cancer classification data separated.xlsx" in which patients' data of normal and 8 different cancer types are provided in different sheets): This code presents the topological data analysis(TDA) results for normal and different cancer types based on different patients' biomarker concentrations obtained from blood tests. It includes the characterization of normal and different cancer types based on TDA features and dissimilarity percentages among their statistics
# Import the required libraries pandas, numpy, Rips, matplotlib.pyplot, gudhi, csv, math
# Step 1: Load the clinical data(clinical cancer classification data separated.xlsx) file contained different sheets of different data types
# Provide the data sheet name from the file to select the required data type(E.g.: Normal, Breast, etc) for analysis
# Step 2: Log Normalization
# Log normalize the loaded data type
# Step 3: Select all 39 biomarkers for analysis by selecting the specific columns from the sheet
# Step 4: Calculate Pearson correlation matrix
# Determine pairwise correlations between the biomarkers for the specific data type and form a correlation matrix
# Step 5: Convert the correlation matrix into a distance matrix
#Step 6: Construct a simplicial complex and apply persistent homology to the distance matrix
# Plot Persistence Diagram(PD) and Persistence Barcode(PB)
# Step 7: Extract Topological features for the specific data type
# Extract Topological features for the specific data type which includes: 1-th Betti Numbers Curve and a feature vector containing statistical measures of the topological features
# Thus extract the topological features for all different data types by providing the sheet or data type name one by one in Step 1
# All the statistics of the topological features of normal and different cancer types are listed in the Table 1 sheet of "clinical cancer classification data separated.xlsx" file
# Compare the 1-th Betti Numbers Curves of different data types
# Step 8: Pairwise dissimilarity percentage calculation between the topological features of normal and different cancer types
# Calculate RMSE between the statistics of topological features of each data type
# Convert the RMSE into dissimilarity percentages
------------------------------------------------------------------------------------------------------------------
# Script2.ipynb(Data file used: "clinical cancer data.xlsx" in which all the patient's data including normal and 8 different cancer types are listed in one sheet(named Normal and Cancer) together). This code presents the topological data analysis(TDA) results for all patients(Training and validation) at an individual level based on their biomarker concentrations obtained from blood tests. It includes two main outcomes: cancer detection and identifying the cancer-origin tissue, specifically whether it is stomach cancer.
# Step 1: Load the clinical data("clinical cancer data.xlsx") file
# Provide the sheet name "Normal and Cancer"
# Step 2: Select the biomarkers
# 'CA-125', 'CA 15-3', 'CA19-9', 'CEA', 'CYFRA 21-1', 'FGF2', 'HE4', 'HGF', 'IL-6', 'IL-8', 'TGFa' are the biomarkers selected to classify cancer patients from normal one and detect Stomach cancer if it presents
# Step 3: Select rows 0 to 59 from the sheet to create and fix a dataset for 59 normal patients
# Step 4:  Add one additional random patient data for analysis to the fixed dataset of 59 normal patients
# Step 5: Log normalize the combined 60 patients' data
# Step 6: Calculate Pearson correlation matrix and convert it into a distance matrix
# Determine pairwise correlations between the selected biomarkers for the combined 60 patients' data and form a correlation matrix
# Convert the correlation matrix into a distance matrix
# Step 7: Construct a simplicial complex and apply persistent homology to the distance matrix
# Plot Persistence Diagram(PD) and Persistence Barcode(PB)
# Step 8: Extract topological features which include statistical measures of 0 and 1-dimensional holes
# The random patients from the sheet were added one by one automatically with the fixed dataset of 59 normal patients and thus the topological features for each random patient are calculated
# Step 9: Save the updated DataFrame, which contains the topological features for all additional random patients, to the CSV file "FeatureVectorStomachDet.csv"
# Step 10: Load the CSV file "FeatureVectorStomachDet.csv"
# Step 11: Construct a regression model to get a regression equation
# Select topological features('Avg_Death_0D', '0-Dim Hole Min Life Range', '0-Dim Hole Max Life Range', 'Median_Death_0D', 'Std_Death_0D') of 500 training normal patients
# Prepare the data for regression
# Select 'Avg_Death_0D' as the response variable and others as predictors
# Determine a regression equation from the provided data
# Step 12: Compare the regression equation with any other random patient's topological features
# Calculate the difference between Actual Avg_Death_0D and predicted Avg_Death_0D of the random patient and get the 'Regression Difference Percentage'
# Step 13: Compare the Average Birth and Average Death of 1 Dimensional holes feature of the random patient with the 500 normal patients' training datasets 
and get the 'Distance %(1)' and additionally compare '1-Dim Hole Max Life Range' to separate cancer patients from normal one
# Step 14: Define the conditions of cancer detection
# Apply thresholds-based conditions for 'Regression Difference Percentage', 'Distance %(1)' and '1-Dim Hole Max Life Range' to classify cancer patients from normal ones
# Step 15: Cancer detection
# If any two conditions are satisfied, then the patient is detected to be cancer-positive and saved as Result(Cancer)
# Step 16: Stomach Cancer Detection
# Step 16(a): Select the topological features '0-Dim Hole Min Life Range' and 'Range_Death_0D' contained in the CSV file "FeatureVectorStomachDet.csv"
# Step 16(b): A reference point in Stomach cancer for the above features is determined by using 50 Stomach cancer patients' training datasets for comparison with the random patient
# The x and y coordinates in the reference point are the averages of '0-Dim Hole Min Life Range' and 'Range_Death_0D' in 50 Stomach cancer patients' training datasets. The z coordinate is the maximum(round off) '0-Dim Hole Min Life Range' observed in the 50 Stomach cancer patients' training datasets
# Step 16(c): Compare the reference point with the random patients' respective topological features and get Distance %(2)
# Step 16(d): Apply threshold-based condition for 'Result(Stomach)' based on 'Distance %(2)'
# Step 16(e): Check if both 'Result(Cancer)' and 'Result(Stomach)' are Positive to detect a Stomach cancer positive
# The code is automatized to detect cancer and stomach cancer if it presents for all the additional random patients(Training and validation) of the data file clinical cancer data.xlsx in one run
# The extracted results are stored in the CSV file "FeatureVectorStomachDet.csv"
------------------------------------------------------------------------------------------------------------------
# Script3.ipynb(Data file used: "clinical cancer data.xlsx" in which all the patient's data including normal and 8 different cancer types are listed in one sheet(named Normal and Cancer) together). This code presents the topological data analysis(TDA) results for all patients(Training and validation) at an individual level based on their biomarker concentrations obtained from blood tests to identify the cancer origin tissue, specifically whether the cancer is pancreatic cancer. This code uses the result 'Result(Cancer)' directly from the Script2.ipynb and then does some additional TDA analysis to identify whether the cancer is pancreatic cancer. Perform Script2.ipynb first to obtain the CSV file 'FeatureVectorStomachDet.csv' and get the result 'Result(Cancer)' for the random patient
# Step 1: Load the clinical data("clinical cancer data.xlsx") file
# Provide the sheet name "Normal and Cancer"
# Step 2: Select the biomarkers
# 'CA19-9', 'CEA', 'TGFa' are the biomarkers selected to classify and detect Pancreas cancer patients from other types of patients if they present
# Step 3: Select rows 0 to 59 from the sheet to create and fix a dataset for 59 normal patients
# Step 4:  Add one additional random patient data for analysis to the fixed dataset of 59 normal patients
# Step 5: Log normalize the combined 60 patients' data
# Step 6: Calculate Pearson correlation matrix and convert it into a distance matrix
# Determine pairwise correlations between the selected biomarkers for the combined 60 patients' data and form a correlation matrix
# Convert the correlation matrix into a distance matrix
# Step 7: Construct a simplicial complex and apply persistent homology to the distance matrix
# Plot Persistence Diagram(PD) and Persistence Barcode(PB)
# Step 8: Extract topological features which include statistical measures of 0 and 1-dimensional holes
# The random patients from the sheet were added one by one automatically with the fixed dataset of 59 normal patients and thus the topological features for each random patient are calculated
# Step 9: Save the updated DataFrame, which contains the topological features for all additional random patients, to the CSV file "FeatureVectorPancreasDet.csv"
# Step 10: Add the 'Result(Cancer)' column containing cancer detection results of the random patients from the existing CSV file 'FeatureVectorStomachDet.csv' to the CSV file 'FeatureVectorPancreasDet.csv' and get the cancer detection result of the random patient
# Step 11: Pancreas Cancer Detection
# Step 11(a): Select the topological features '0-Dim Hole Min Life Range' and 'Range_Death_0D' contained in the CSV file "FeatureVectorPancreasDet.csv"
# Step 11(b): A reference point in Pancreas cancer for the above features is determined by using 69 Pancreas cancer patients' training datasets for comparison with the random patient
# The x and y coordinates in the reference point are the averages of '0-Dim Hole Min Life Range' and 'Range_Death_0D' in 69 Stomach cancer patients' training datasets.
# Step 11(c): Compare the reference point with the random patients' respective topological features and get Distance %(3)
# Step 11(d): Apply threshold-based condition for 'Result(Pancreas)' based on 'Distance %(3)'
# Step 11(e): Check if both 'Result(Cancer)' and 'Result(Pancreas)' are Positive to detect a Pancreas cancer positive
# The code is automatized to detect Pancreas cancer if it presents for all the additional random patients(Training and validation) of the data file clinical cancer data.xlsx in one run
# The extracted results are stored in the CSV file "FeatureVectorPancreasDet.csv"