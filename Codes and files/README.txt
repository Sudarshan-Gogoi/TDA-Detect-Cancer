# README for Topological Data Analysis (TDA) for Cancer Type Characterization (Characterization.ipynb):

## Overview
This Jupyter Notebook (Characterization.ipynb) performs topological data analysis on clinical cancer biomarker data to characterize and differentiate between normal tissue and 8 cancer types using persistent homology and topological feature extraction.

## Key Features
- Processes clinical biomarker concentration data from blood tests
- Computes topological invariants for cancer characterization
- Generates persistence diagrams and barcode visualizations
- Extracts quantitative topological features
- Calculates pairwise dissimilarity between cancer types
- Comparative analysis of 1th Betti numbers curves

## Data Requirements
- Input file: 'clinical cancer classification data separated.xlsx'
  - Contains 9 sheets:
    - Normal tissue data
    - 8 cancer type datasets (Breast, Colorectal, etc.)
    - Table 1: Precomputed topological feature statistics

## Analysis Pipeline
1. **Data Loading**
   - Select dataset by sheet name (e.g., 'Normal', 'Breast')
   
2. **Preprocessing**
   - Log-normalization of biomarker concentrations
   - Selection of 39 biomarkers for analysis

3. **Topological Analysis**
   - Pearson correlation matrix computation
   - Distance matrix conversion
   - Simplicial complex construction
   - Persistent homology calculation

4. **Visualization**
   - Persistence diagrams
   - Barcode plots
   - 1D Betti number curves

5. **Feature Extraction**
   - Statistical measures of topological features:
     - 0D/1D hole life ranges
     - Birth/death averages
     - Hole counts
     - Distribution statistics

6. **Comparative Analysis**
   - Pairwise dissimilarity percentage calculations
   - RMSE between topological feature vectors
   - Comparative Betti curve analysis

## Usage Instructions

### Dependencies
- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - ripser
  - gudhi
  - matplotlib
  - scipy (for distance calculations)

### Running the Analysis
1. Configure the input parameters:
   ```python
   file_path = 'path/to/clinical cancer classification data separated.xlsx'
   sheet_name = 'Stomach'  # Change to desired tumor type
-------------------------------------------------------------------------------------------------------------

# README for Biomarker Selection algorithm (Optimized_Biomarker_Selection.ipynb):
# Optimized_Biomarker_Selection function

## Overview
The Optimized_Biomarker_Selection algorithm is a Python-based pipeline designed to identify a minimal set of significant biomarkers for cancer detection using Topological Data Analysis (TDA) and Linear Regression. The algorithm processes clinical cancer data, extracts topological features, builds predictive models, and iteratively eliminates the least significant biomarkers to optimize the selection process. The goal is to maximize sensitivity while maintaining a fixed specificity of 95%. The algorithm outputs a list of significant biomarkers and saves intermediate results for further analysis.

## Key Concepts
1. **Sensitivity Analysis**: The algorithm evaluates the sensitivity of cancer detection using TDA-based features. Sensitivity is calculated for different sets of biomarkers, and the goal is to maximize it while keeping specificity fixed at 95%.
2. **Minimal-Impact Biomarkers**: Biomarkers whose removal increases sensitivity are considered "minimal-impact" and are eliminated in the iterative process.
3. **Maximal-Impact Biomarkers**: Biomarkers whose removal decreases sensitivity are considered "maximal-impact" and are retained for further analysis.
4. **Iterative Elimination**: The algorithm iteratively eliminates a set of minimal-impact biomarkers to reduce the total number of biomarkers and improve sensitivity.
5. **Distance %(1) and Regression Difference Percentage**: These are key features constructed using TDA. They play a critical role in distinguishing cancer patients from healthy individuals and are adjusted to maintain specificity at 95%.

## Steps Utilized in Optimized_Biomarker_Selection

### Part 1: Perform TDA on Clinical Data
1. **Data Loading**  
   - Load clinical cancer data from an Excel file (clinical_data_path) and a specific sheet (Normal and Cancer).  
   - Two DataFrames (df_1 and df_2) are created to store the data.

2. **Biomarker Selection**  
   - The user specifies a list of biomarkers (Select_Biomarkers) to analyze.  
   - The first 59 rows of df_1 are used as a fixed dataset of normal patients.  
   - df_2 contains the full dataset for analysis.

3. **Topological Data Analysis (TDA)**  
   - For each patient (from row 60 to max_patients):  
     - A new dataset is created by appending the patient's data to the fixed dataset of 59 normal patients.  
     - Numeric columns are log-normalized.  
     - A Pearson correlation matrix is computed and converted into a distance matrix.  
     - A Rips complex is constructed, and persistence homology is calculated using the gudhi library.  
     - Topological features (0D and 1D holes) are extracted, including:  
       - Life ranges (min, max, average, median, standard deviation).  
       - Number of holes.  
       - Birth and death values of holes.

4. **Feature Vector Creation**  
   - The extracted topological features are stored in a DataFrame (data_list).  
   - Additional patient information (e.g., Patient ID, Sample ID, Cohort, AJCC Stage) is appended to the feature vector.

5. **Save Results to CSV**  
   - The combined feature vector and patient information are saved to a CSV file (feature_vector_path).

### Part 2: Construct a Linear Regression Model
1. **Model Training**  
   - A linear regression model is trained using the normal patient data (first normal_data_size rows).  
   - The model predicts the average death of 0D holes (Avg_Death_0D) based on selected features:  
     - 0-Dim Hole Min Life Range  
     - 0-Dim Hole Max Life Range  
     - Median_Death_0D  
     - Std_Death_0D

2. **Prediction and Analysis**  
   - The trained model is used to predict Avg_Death_0D for all patients.  
   - The difference between actual and predicted values is calculated.  
   - A reference point is computed as the mean of Average Birth (1D) and Average Death (1D) for the normal cohort.  
   - Euclidean distance from the reference point is calculated for each patient.  
   - Patients are classified as "Positive" or "Negative" for cancer based on user-defined thresholds:  
     - k_1: Regression difference percentage threshold.  
     - k_2: Distance threshold.

3. **Determine Optimal Thresholds**  
   - The optimal thresholds (k_1 and k_2) are determined by iterating through possible values until the number of "Positive" classifications for the normal cohort matches a predefined value (e.g., 36).

4. **Save Updated Results**  
   - The updated DataFrame with predictions and classifications is saved back to the CSV file.

### Part 3: Leave-One-Out Biomarker Analysis
1. **Leave-One-Out Approach**  
   - For each biomarker in Select_Biomarkers:  
     - Construct a new dataset by leaving out the current biomarker.  
     - Repeat the TDA process (as in Part 1) on the modified dataset.  
     - Save the resulting feature vector to a new CSV file (FeatureVectorCancerDet{i+1}.csv).

2. **Map Biomarkers to CSV Files**  
   - Create a mapping of biomarkers to their respective CSV files for easy reference.

### Part 4: Apply Regression Model to Modified Datasets
1. **Model Application**  
   - For each modified dataset (with one biomarker left out):  
     - Apply the linear regression model (as in Part 2).  
     - Predict and classify patients.  
     - Count the number of positive cancer classifications.

2. **Identify Significant Biomarkers**  
   - Sort the datasets based on the number of positive cancer classifications.  
   - Eliminate the top top_n_files_to_eliminate least significant biomarkers.
   - Avoid eliminating biomarkers that are already validated in existing literature to ensure scientific relevance and maintain the integrity of the biomarker set.
   - Output the remaining significant biomarkers.

### Part 5: Iterative Elimination Process
1. **Iterative Elimination**  
   - Perform multiple iterations of the leave-one-out approach, eliminating a set of minimal-impact biomarkers in each iteration.  
   - Adjust the Distance %(1) threshold in each iteration to maintain specificity at 95%.  
   - Continue the process until the desired sensitivity is achieved.

2. **Final Biomarker Selection**  
   - After multiple iterations, a final set of significant biomarkers is selected.  
   - These biomarkers are validated through a literature survey to confirm their scientific relevance and association with cancer.

## Example Usage
```python
Optimized_Biomarker_Selection(
    distance_threshold_offset=0.6,
    tumor_types={'normal': 554, 'breast': 156, 'colorectum': 291, 'esophagus': 34, 'liver': 33, 'lung': 78, 'ovary': 40, 'pancreas': 69, 'stomach': 50},
    max_patients=1802,
    normal_data_size=554,
    top_n_files_to_eliminate=5,
    cancer_types=['Breast', 'Colorectum', 'Esophagus', 'Liver', 'Lung', 'Ovary', 'Pancreas', 'Stomach'],
    Select_Biomarkers=['Angiopoietin-2', 'CA-125', 'CA 15-3', 'CA19-9', 'CD44', 'CEA', 'CYFRA 21-1', 'FGF2', 'G-CSF', 'GDF15', 'HE4', 'HGF', 'IL-6', 'Myeloperoxidase', 'OPN', 'PAR', 'sHER2/sEGFR2/sErbB2', 'sPECAM-1', 'Thrombospondin-2'],
    clinical_data_path=r'path to the file\Clinical cancer data.xlsx',
    feature_vector_path="path to the file/FeatureVectorCancerDet.csv",
    output_folder_path="path to the output files"
)

## Output Files
1. FeatureVectorCancerDet.csv  
   - Contains topological features and patient information for the full dataset.

2. FeatureVectorCancerDet{i+1}.csv  
   - Contains topological features and patient information for datasets with one biomarker left out.

3. Significant Biomarkers  
   - A list of significant biomarkers after eliminating the least significant ones.

## Dependencies
- Python 3.x  
- Libraries:  
  - pandas  
  - numpy  
  - gudhi  
  - sklearn.linear_model.LinearRegression  
  - os

## Notes
- Ensure the input Excel file has the correct sheet name (Normal and Cancer) and columns.  
- Adjust user-defined variables (e.g., Select_Biomarkers, top_n_files_to_eliminate, thresholds) based on your dataset and requirements.  
- The algorithm is designed for significant biomarker selection
-----------------------------------------------------------------------------------------------------------

# README for Cancer Detection Threshold Optimization (Optimized_Biomarker_Selection.ipynb)
# get_threholds function

## Overview
This algorithm calculate optimal thresholds for cancer detection using:
- "FeatureVectorCancerDet.csv" file obtained from the previous algorithm (Optimized_Biomarker_Selection)
- Linear regression modeling
- Iterative threshold optimization

## Key Features
1. **Threshold Calculation**:
   - Distance threshold (k₂)
   - Regression difference threshold (k₁)
   - 1-Dim Hole Max Life Range

2. **Optimization Process**:
   - Tests multiple threshold offsets (0-2 in 0.1 increments)
   - Tracks best performing parameters
   - Maintains 95% specificity (36/739 normal positives)

3. **Output**:
   - Optimal threshold values
   - Total positive cancer detections
   - Diagnostic statistics

## Usage Example
```python
# Configuration
params = {
    "feature_vector_path": "FeatureVectorCancerDet.csv",
    "normal_data_size": 554,
    "cancer_types": ["Breast", "Colorectum", ...],
    "tumor_types": {"normal": 554},
    "required_normal_positives": 36
}

# Run optimization
find_optimal_thresholds(**params)

## Notes

- Maintains scientific validity by using literature-validated biomarkers and preserving key topological features
- Thresholds are dataset-dependent
-----------------------------------------------------------------------------------------------------------

# README for Cancer Detection Function (TDAcancer_detect_1.ipynb):
# TDAcancer_detect_1 Function

## Overview
The TDAcancer_detect_1 function is a Python-based pipeline designed to detect cancer using Topological Data Analysis (TDA) and machine learning. It processes clinical cancer data, extracts topological features, builds a predictive model, and classifies patients as "Positive" or "Negative" for cancer based on user-defined thresholds. This document explains the steps utilized in the function.

## Steps Utilized in TDAcancer_detect_1
1. Data Loading  
   - The function loads clinical cancer data from an Excel file (file_path).  
   - The data is read from a specific sheet (Normal and Cancer).  
   - Two DataFrames (df_1 and df_2) are created to store the data.

2. Biomarker Selection  
   - The user specifies a list of biomarkers (biomarkers) to analyze.  
   - The first 59 rows of df_1 are used as a fixed dataset of normal patients.  
   - df_2 contains the full dataset for analysis.

3. Topological Data Analysis (TDA)  
   - For each patient (from row 60 to random_patient_range):  
     - A new dataset is created by appending the patient's data to the fixed dataset of 59 normal patients.  
     - Numeric columns are log-normalized.  
     - A Pearson correlation matrix is computed and converted into a distance matrix.  
     - A Rips complex is constructed, and persistence homology is calculated using the gudhi library.  
     - Topological features (0D and 1D holes) are extracted, including:  
       - Life ranges (min, max, average, median, standard deviation).  
       - Number of holes.  
       - Birth and death values of holes.

4. Feature Vector Creation  
   - The extracted topological features are stored in a DataFrame (data_list).  
   - Additional patient information (e.g., Patient ID, Sample ID, Cohort, AJCC Stage) is appended to the feature vector.

5. Save Results to CSV  
   - The combined feature vector and patient information are saved to a CSV file (_features.csv).

6. Linear Regression Model  
   - A linear regression model is trained using the control data (first control_data_range rows).  
   - The model predicts the average death of 0D holes (Avg_Death_0D) based on selected features:  
     - 0-Dim Hole Min Life Range  
     - 0-Dim Hole Max Life Range  
     - Median_Death_0D  
     - Std_Death_0D

7. Prediction and Classification  
   - The trained model is used to predict Avg_Death_0D for all patients.  
   - The difference between actual and predicted values is calculated.  
   - A reference point is computed as the mean of Average Birth (1D) and Average Death (1D) for the control data.  
   - Euclidean distance from the reference point is calculated for each patient.  
   - Patients are classified as "Positive" or "Negative" for cancer based on user-defined thresholds obtained from the previous algorithm (get_thresholds)
     - k_1: Regression difference percentage threshold.  
     - k_2: Distance threshold.  
     - k_3: Upper threshold for 1D hole max life range.  
     - k_4: Lower threshold for 1D hole max life range.  
   - A patient is classified as "Positive" if at least two of the four conditions are met.

8. Save Updated Results  
   - The updated DataFrame with predictions and classifications is saved back to the CSV file.

9. Count Positive Results  
   - The number of "Positive" classifications is counted for each cohort (e.g., Normal, Breast, Colorectum).  
   - The total number of positive cancer cases (excluding "Normal") is calculated.

10. Return Results  
    - The function returns a dictionary containing:
      - Positive counts for each tumor type.  
      - Total positive cancer cases.

## Example Usage
file_path = r'path to the file\Clinical cancer data.xlsx'

# User-defined variables
biomarkers = ['Angiopoietin-2', 'CA-125', 'CA 15-3', 'CEA', 'CYFRA 21-1', 'FGF2', 'G-CSF', 'HE4', 'HGF', 'PAR', 'sPECAM-1', 'Thrombospondin-2']
random_patient_range = 1802  # User-defined range
control_data_range = 554  # User-defined control data range
k_1 = 0.4440  # Regression threshold
k_2 = 2.3  # Distance threshold
k_3 = 0.112  # Upper threshold for 1D hole max life range
k_4 = 0.047  # Lower threshold for 1D hole max life range
tumor_types = ['Normal', 'Breast', 'Colorectum', 'Esophagus', 'Liver', 'Lung', 'Ovary', 'Pancreas', 'Stomach']

# Run the pipeline
results = TDAcancer_detect_1(
    file_path, biomarkers, random_patient_range, control_data_range, 
    k_1, k_2, k_3, k_4, tumor_types
)
print(results)

## Output
The function outputs a dictionary with the following keys:  
- "<Tumor Type> Positive": Number of positive cases for each tumor type.  
- "Total Positive Cancers": Total number of positive cancer cases (excluding "Normal").

### Example output:
{
    "Normal Positive": 36,
    "Breast Positive": 77,
    "Colorectum Positive": 182,
    "Esophagus Positive": 25,
    "Liver Positive": 29,
    "Lung Positive": 47,
    "Ovary Positive": 41,
    "Pancreas Positive": 42,
    "Stomach Positive": 55,
    "Total Positive Cancers": 498
}

## Dependencies
- Python 3.x  
- Libraries:  
  - pandas  
  - numpy  
  - gudhi  
  - scikit-learn (sklearn.linear_model.LinearRegression)

## Notes
- Ensure the input Excel file has the correct sheet name (Normal and Cancer) and columns.  
- Adjust user-defined variables (e.g., biomarkers, thresholds) based on your dataset and requirements.  
- The function is designed for cancer detection but can be adapted for other applications by modifying the biomarkers and thresholds.
----------------------------------------------------------------------------------------------------------

# README for Cancer Tissue Localization Function (TDAcancer_detect_2.ipynb):
# TDAcancer_detect_2 Function

## Overview
The TDAcancer_detect_2 function is a Python-based pipeline designed to localize cancer tissue using Topological Data Analysis (TDA) and machine learning. It processes clinical cancer data, extracts topological features, and builds predictive models to classify cancer tissue as "Positive" or "Negative" based on different biomarker combinations. The function also evaluates the sensitivity and accuracy of these biomarkers in detecting specific cancer types. Additionally, it generates output files for further analysis, helping identify the biomarkers that offer the highest sensitivity and accuracy for cancer detection.

## Steps Utilized in TDAcancer_detect_2

### Part 1: Preliminary Analysis and Formation of Average1.xlsx
1. Data Loading  
   - Load clinical cancer data from an Excel file (file_path) and a specific sheet (sheet_name).  
   - Two DataFrames (df_1 and df_2) are created to store the data.

2. Biomarker Selection  
   - The user specifies a list of biomarkers (biomarkers) to analyze.  
   - The first 59 rows of df_1 are used as a fixed dataset of normal patients.  
   - df_2 contains the full dataset for analysis.

3. Generate Biomarker Combinations  
   - All possible combinations of 3 biomarkers are generated using itertools.combinations.  
   - The number of combinations to process is user-defined (num_combinations).

4. Topological Data Analysis (TDA)  
   - For each combination of biomarkers:  
     - A new dataset is created by appending patient data to the fixed dataset of 59 normal patients.  
     - Numeric columns are log-normalized.  
     - A Pearson correlation matrix is computed and converted into a distance matrix.  
     - A Rips complex is constructed, and persistence homology is calculated using the gudhi library.  
     - Topological features (0D and 1D holes) are extracted, including:  
       - Life ranges (min, max, average, median, standard deviation).  
       - Number of holes.  
       - Birth and death values of holes.

5. Feature Vector Creation  
   - The extracted topological features are stored in a DataFrame (data_list).  
   - Additional patient information (e.g., Patient ID, Sample ID, Cohort, AJCC Stage) is appended to the feature vector.

6. Save Results to CSV  
   - The combined feature vector and patient information are saved to a CSV file (FeatureVectorTissueDet{i}.csv).

7. Calculate Averages  
   - For each combination, cohort averages are calculated for specific features (e.g., 0-Dim Hole Min Life Range, Range_Death_0D).  
   - Euclidean distances from a reference point (mean of normal cohort) are computed.  
   - A threshold (k) is determined based on the second-highest distance.

8. Classify Cancer Tissue  
   - Patients are classified as "Positive" or "Negative" for cancer tissue based on the distance threshold.  
   - Results are saved back to the CSV file.

9. Merge Averages to Excel  
   - Averages from all combinations are merged into a single Excel file (Average1.xlsx).

### Part 2: Formation of Positives1.xlsx and Sensitivity Outcomes
1. Count Positives  
   - For each combination, count the number of "Positive" classifications for each cohort and AJCC stage.  
   - Save results to CSV files (positives_countDet{i}.csv).

2. Merge Positives to Excel  
   - Merge all positives_countDet{i}.csv files into a single Excel file (Positives1.xlsx).

3. Calculate Sensitivity and Accuracy  
   - Sensitivity and accuracy are calculated for each combination based on true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN).  
   - Results are saved back to Positives1.xlsx.

4. Get Top 5 Sensitivities  
   - For each cohort, identify the top 5 biomarker combinations with the highest sensitivity and accuracy (> 60%).  
   - Save results to an Excel file (Top_5_Sensitivities.xlsx).

## Example Usage
file_path = r'path to the file\Clinical cancer data.xlsx'
sheet_name = 'Normal and Cancer'
biomarkers = ['Angiopoietin-2', 'CA-125', 'CA 15-3', 'CEA', 'CYFRA 21-1', 'FGF2', 'G-CSF', 'HE4', 'HGF', 'PAR', 'sPECAM-1', 'Thrombospondin-2']
output_dir = r"output file path"

# User-defined inputs
tumor_types = {
    'normal': 528,
    'breast': 58,
    'colorectum': 137,
    'esophagus': 18,
    'liver': 22,
    'lung': 35,
    'ovary': 31,
    'pancreas': 32,
    'stomach': 41
}
total_positive_patients = {
    "Breast": 77, "Colorectum": 182, "Esophagus": 25, "Liver": 29,
    "Lung": 47, "Ovary": 41, "Pancreas": 42, "Stomach": 55
}
cohorts_to_analyze = ['Breast', 'Colorectum', 'Esophagus', 'Liver', 'Lung', 'Ovary', 'Pancreas', 'Stomach']
num_combinations = 220  # User-defined number of combinations
random_patient_range = 1802  # User-defined random patient range

# Execute the TDA_cancerDetect function
TDAcancer_detect_2(file_path, sheet_name, biomarkers, output_dir, tumor_types, total_positive_patients, num_combinations, random_patient_range, cohorts_to_analyze)

## Output Files
1. FeatureVectorTissueDet{i}.csv  
   - Contains topological features and patient information for each biomarker combination.

2. Average1.xlsx  
   - Contains average distances and other metrics for each biomarker combination.

3. Positives1.xlsx  
   - Contains positive counts, sensitivity, and accuracy for each cohort and biomarker combination.

4. Top_5_Sensitivities.xlsx  
   - Contains the top 5 biomarker combinations with the highest sensitivity and accuracy for each cohort.

## Dependencies
- Python 3.x  
- Libraries:  
  - pandas  
  - numpy  
  - gudhi  
  - itertools  
  - re  
  - os

## Notes
- Make the input and output file paths for the two functions TDAcancer_detect_1 and TDAcancer_detect_2 same because the files are interconnected to each other.
- Ensure the input Excel file has the correct sheet name (Normal and Cancer) and columns.
- Adjust user-defined variables (e.g., biomarkers, num_combinations, thresholds) based on your dataset and requirements.  
- The function is designed for cancer tissue localization but can be adapted for other applications by modifying the biomarkers and thresholds.
-----------------------------------------------------------------------------------------------------------
