# TDAcancer-detect README: Clinical Cancer Detection & Tissue Localization
Example dataset included: clinical cancer validation data.xlsx
------------------------------------------------------------------------------------

# Optimized Biomarker Selection Algorithm ('Optimized_Biomarker_Selection' function)

## Overview
This algorithm identifies the most significant biomarkers for cancer detection by analyzing protein expression data. It uses topological data analysis and machine learning to recursively eliminate less important biomarkers and select the most impactful ones.

## How It Works

### 1. Data Processing
- Reads clinical data from Excel files
- Processes patient data starting from row 60
- Handles multiple cancer types (bladder, ovarian, pancreatic) and normal samples
- Uses logarithmic transformation of biomarker values

### 2. Topological Data Analysis (TDA)
- Calculates correlation matrices between biomarkers
- Creates distance matrices (1 - absolute correlation)
- Uses Gudhi library to perform persistent homology analysis
- Extracts features like:
  - 0-dimensional and 1-dimensional holes in data
  - Life ranges of topological features
  - Birth and death values of topological features

### 3. Machine Learning Model
- Builds linear regression model using topological features
- Predicts average death values for 0-dimensional holes
- Calculates regression differences and distances from normal reference points

### 4. Cancer Detection Criteria
A patient is classified as cancer-positive if BOTH conditions are met:
1. Regression difference percentage exceeds threshold k₁
2. Distance from normal reference point exceeds threshold k₂

### 5. Biomarker Selection Process
- Tests each biomarker by temporarily removing it
- Ranks biomarkers by their impact on cancer detection accuracy
- Eliminates the least significant biomarkers
- Returns the most important biomarkers for cancer detection

## Input Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `distance_threshold_offset` | Offset for distance threshold calculation | 1.6 |
| `tumor_types` | Dictionary of cancer types and patient counts | `{'normal': 50, 'bladder': 19, ...}` |
| `max_patients` | Maximum number of patients to process | 322 |
| `normal_data_size` | Number of normal patients for training | 50 |
| `top_n_files_to_eliminate` | Number of biomarkers to eliminate each iteration | 2 |
| `cancer_types` | List of cancer types to consider | `['bladder', 'ovarian', 'pancreatic']` |
| `Select_Biomarkers` | List of all biomarkers to analyze | `['Tenascin C', 'sAXL', ...]` |
| `clinical_data_path` | Path to Excel file with clinical data | `D:\Cancer Detection folder\...` |
| `sheet_name` | Excel sheet name | `'Normal and Cancer'` |
| `feature_vector_path` | Output path for feature vectors | `D:/Cancer Detection folder/...` |
| `output_folder_path` | Folder for output files | `D:/Cancer Detection folder/...` |
| `FPR` | False positive rate target | 0.05 |

## Output
- CSV files with topological features for each biomarker combination
- List of most significant biomarkers for cancer detection
- Performance metrics for each biomarker subset

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, gudhi libraries
- Excel file with clinical data in specified format

## Usage
1. Prepare clinical data in Excel format with required columns
2. Set appropriate file paths in the parameters
3. Run the 'Optimized_Biomarker_Selection' function
4. Review output files and selected biomarkers

The algorithm will automatically process the data, perform analysis, and identify the optimal set of biomarkers for cancer detection. Repeat the same process with the newly obtained optimal biomarker set if it doesn't provide a satisfactory overall detection outcome.
------------------------------------------------------------------------------------------------------------------------------------------------------

# Get Thresholds Algorithm - Part 1 (get_thresholds_1 function)

## Overview
This algorithm calculates the optimal thresholds (k₁ and k₂) for cancer detection using selected biomarkers. It fine-tunes these thresholds to maximize cancer detection while maintaining high specificity (above 95%).

## How It Works

### 1. Data Loading
- Reads the feature vector CSV file created by the biomarker selection algorithm
- Contains topological features calculated from biomarker data

### 2. Regression Model
- Builds a linear regression model using four topological features:
  - 0-Dimensional Hole Min Life Range
  - 0-Dimensional Hole Max Life Range  
  - Median Death (0D)
  - Standard Deviation of Death (0D)
- Predicts Average Death values for 0-dimensional holes
- Calculates regression differences between actual and predicted values

### 3. Reference Point Calculation
- Determines a normal reference point using average birth and death values from healthy patients
- Calculates Euclidean distances of all patients from this normal reference point

### 4. Threshold Optimization
- Tests different threshold values to find the optimal balance between:
  - Sensitivity (cancer detection rate)
  - Specificity (minimizing false positives in normal patients)
- A patient is classified as cancer-positive if BOTH conditions are met:
  1. Regression difference percentage > threshold k₁
  2. Distance from normal reference > threshold k₂

### 5. Iterative Tuning
- Automatically tests multiple distance threshold offsets
- Selects the configuration that gives the highest cancer detection rate
- Maintains specificity above 95% by controlling false positives in normal samples

## Input Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `feature_vector_path` | Path to feature vector CSV file | `D:/Cancer Detection folder/...` |
| `normal_data_size` | Number of healthy patients for training | 50 |
| `distance_threshold_offset` | Offset for distance threshold calculation | 0.8 to 2.0 |
| `cancer_types` | Cancer types to count positives | `['Bladder', 'Ovarian', 'Pancreatic']` |
| `tumor_types` | Dictionary of tumor types | `{'normal': 50}` |
| `required_normal_positives` | Maximum allowed false positives | 2 |

## Output
- Optimal threshold k₁ (Regression Difference Threshold)
- Optimal threshold k₂ (Normal Distance Threshold) 
- Total number of cancer positives detected
- 1-Dimensional Hole Max Life Range for normal cohort

## Key Features
- **Automatic Tuning**: Tests multiple threshold configurations automatically
- **Specificity Control**: Ensures false positive rate stays below 5%
- **Performance Optimization**: Selects thresholds that maximize cancer detection
- **Reproducible**: Consistent results with same input parameters

## Usage
1. First run the Biomarker Selection Algorithm to get the feature vector file
2. Set the path to the feature vector CSV file
3. Specify the number of normal patients for calibration
4. Define the required specificity level (via required_normal_positives)
5. Run the algorithm to get optimal thresholds k₁ and k₂

The algorithm will output the best thresholds that maximize cancer detection while maintaining high specificity.
------------------------------------------------------------------------------------------------------------------------------------------------------

# Get Thresholds Algorithm - Part 2 (get_thresholds_2 function)

## Overview
This algorithm calculates two additional thresholds (k₃ and k₄) for cancer detection that work alongside the previously determined k₁ and k₂ thresholds. These new thresholds focus on the 1-Dimensional Hole Max Life Range feature to further refine cancer detection accuracy.

## How It Works

### 1. Data Preparation
- Loads the feature vector CSV file created earlier
- Uses control data (normal patients) to train the regression model

### 2. Regression Model
- Uses the same linear regression model as previous steps
- Predicts Average Death values for 0-dimensional holes
- Calculates regression differences between actual and predicted values

### 3. Additional Threshold Calculation
- Uses the 1-Dimensional Hole Max Life Range feature
- Sets two thresholds (k₃ and k₄) that create a range around the normal value
- k₃ = normal value + adjustment (upper bound)
- k₄ = normal value - adjustment (lower bound)

### 4. Iterative Threshold Tuning
- Starts with small adjustments to the normal value
- Gradually increases the adjustment range
- Stops when the number of false positives in normal patients meets the target
- Ensures specificity remains high by controlling false positives

### 5. Final Cancer Detection
A patient is classified as cancer-positive if AT LEAST 2 of these 4 conditions are met:
1. Regression difference percentage > k₁
2. Distance from normal reference > k₂  
3. 1-Dim Hole Max Life Range > k₃ (above upper bound)
4. 1-Dim Hole Max Life Range < k₄ (below lower bound)

## Input Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `feature_vector_path` | Path to feature vector CSV | `D:/Cancer Detection folder/...` |
| `random_patient_range` | Total patients to analyze | 322 |
| `control_data_range` | Normal patients for training | 50 |
| `regression_threshold` | Previously found k₁ value | 0.286 |
| `distance_threshold` | Previously found k₂ value | 1.63 |
| `tumor_types` | Cancer types to track | `['Bladder', 'Ovarian', 'Pancreatic']` |
| `normal_1D_hole_max_life_range` | Normal reference value | 0.1026 |
| `normal_positive_required` | Target false positives | 3 |

## Output
- Final k₃ threshold (upper bound)
- Final k₄ threshold (lower bound) 
- Cancer detection counts for each tumor type
- Total cancer positives detected
- Number of false positives in normal patients
- Final adjustment value used

## Key Features
- **Four-Parameter System**: Uses k₁, k₂, k₃, k₄ for comprehensive detection
- **Range-Based Thresholding**: k₃ and k₄ create an exclusion zone around normal values
- **Adaptive Tuning**: Automatically finds optimal threshold range
- **High Specificity**: Controls false positives to maintain accuracy
- **Flexible Criteria**: Requires only 2 out of 4 conditions to be positive

## Usage
1. First run the previous threshold algorithms to get k₁, k₂, and normal reference values
2. Provide the path to your feature vector file
3. Set the previously determined threshold values
4. Specify the normal reference value for 1-Dim Hole Max Life Range
5. Define the maximum allowed false positives in normal patients
6. Run the algorithm to get optimal k₃ and k₄ values

The algorithm outputs the final thresholds and detection results for all cancer types.
--------------------------------------------------------------------------------------------------------------------------------------------------

# Cancer Detection Module

## Main Function
**TDAcancer_detect_1()** - Main function that executes the complete cancer detection pipeline

## Overview
This is the main cancer detection function that applies the complete pipeline to detect cancer in patients. It combines topological data analysis with machine learning and uses the optimized thresholds (k₁, k₂, k₃, k₄) to classify patients as cancer-positive or negative.

## How It Works

### 1. Data Loading and Preparation
- Loads clinical Excel data from specified file path
- Selects the optimized biomarkers identified in previous steps
- Uses first 59 patients as fixed normal reference dataset

### 2. Topological Data Analysis (TDA)
- Processes each patient individually (from row 60 onwards)
- Combines patient data with the 59 normal references
- Applies logarithmic normalization to biomarker values
- Calculates Pearson correlation matrices
- Converts correlations to distance matrices (1 - absolute correlation)
- Uses Gudhi library to compute persistent homology

### 3. Feature Extraction
Extracts topological features from the analysis:
- **0-Dimensional Features**: Life ranges, death statistics, hole counts
- **1-Dimensional Features**: Birth/death averages, maximum life range, hole counts

### 4. Machine Learning Prediction
- Trains linear regression model on control data (normal patients)
- Predicts expected values for topological features
- Calculates regression differences between actual and predicted values
- Computes distances from normal reference point

### 5. Cancer Classification
A patient is classified as cancer-positive if AT LEAST 2 of these 4 conditions are met:
1. Regression difference percentage > k₁
2. Distance from normal reference > k₂
3. 1-Dim Hole Max Life Range > k₃ (above upper bound)
4. 1-Dim Hole Max Life Range < k₄ (below lower bound)

### 6. Results Compilation
- Saves all topological features and results to CSV file
- Counts positive detections for each cancer type
- Calculates total cancer positives across all types

## Input Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `file_path` | Path to clinical Excel file | `D:\Cancer Detection folder\...` |
| `biomarkers` | Optimized biomarker list | `['sc-kit/SCFR', 'CA-125', ...]` |
| `random_patient_range` | Total patients to analyze | 322 |
| `control_data_range` | Normal patients for training | 50 |
| `k_1` | Regression difference threshold | 0.286 |
| `k_2` | Distance threshold | 1.63 |
| `k_3` | Upper bound for 1D hole range | 0.1186 |
| `k_4` | Lower bound for 1D hole range | 0.0866 |
| `tumor_types` | Cancer types to track | `['Normal', 'Bladder', ...]` |

## Output
- CSV file with all topological features and detection results
- Count of positive detections for each cancer type
- Total number of cancer-positive patients detected
- Complete feature dataset for further analysis

## Key Features
- **Complete Pipeline**: Handles data loading, TDA, ML, and classification
- **Individual Patient Analysis**: Processes each patient separately against normal references
- **Comprehensive Feature Extraction**: Calculates 11 topological features
- **Four-Threshold System**: Uses k₁, k₂, k₃, k₄ for robust classification
- **Results Export**: Saves complete dataset with detection outcomes

## Usage
1. First run all previous algorithms to get optimized biomarkers and thresholds
2. Provide the path to your clinical Excel file
3. Specify the optimized biomarker list
4. Set the previously determined threshold values (k₁, k₂, k₃, k₄)
5. Define the patient range and cancer types to analyze
6. Run the **TDAcancer_detect_1()** function to get detection results

The algorithm outputs detection counts for each cancer type and saves a complete feature dataset for further analysis.
--------------------------------------------------------------------------------------------------------------------------------------------------------

# Tissue of Origin Localization Module

## Purpose
This module is specifically designed to evaluate tissue localization performance on the **local calibration sample** of the dataset. The primary objective is to identify the most significant biomarker combinations that provide optimal cancer tissue identification for different cancer types (Bladder, Ovarian, Pancreatic).

## Overview
This analysis is performed on the calibration sample to:
- Determine which 3-biomarker combinations work best for tissue-specific identification
- Establish optimal threshold parameters for each cancer type
- Identify biomarker sets with the highest localization accuracy
- Provide a performance baseline before full dataset application

## Main Function
**TDAcancer_detect_2_LC()** - Executes the complete tissue localization pipeline on calibration data

## How It Works

### Calibration Sample Analysis
- Uses a subset of the full dataset for initial testing and optimization
- Processes clinical data with different biomarker combinations (3 biomarkers at a time)
- Performs topological data analysis (TDA) on each combination
- Calculates performance metrics specifically for the local calibration sample

### Performance Evaluation
- Tests 40 different threshold offsets (-20 to +19) on the calibration data
- Applies offset thresholds to distance metrics
- Calculates cancer tissue detection accuracy for each offset
- Computes performance metrics (Sensitivity, Precision, F1 Score, Accuracy)

### Significant Biomarker Identification
- Analyzes all results to find top-performing biomarker combinations
- Ranks combinations by F1 scores for each cancer type
- Identifies optimal offsets for each tissue type on the calibration sample
- Selects the most significant biomarker sets for future full-dataset application

## Input Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `file_path` | Clinical Excel file path | `D:\Cancer Detection folder\...` |
| `sheet_name` | Excel sheet name | `'Normal and Cancer'` |
| `biomarkers` | Biomarkers to test combinations | `['sc-kit/SCFR', 'CA-125', ...]` |
| `output_dir` | Output directory | `D:\Cancer Detection folder\Test 3` |
| `tumor_types` | Patient counts by type in calibration sample | `{'normal': 47, 'bladder': 9, ...}` |
| `total_positive_patients` | Cancer patients by type in calibration sample | `{"Bladder": 9, "Ovarian": 11, ...}` |
| `num_combinations` | Number of 3-biomarker combinations to test | 56 |
| `random_patient_range` | Total patients in calibration sample | 322 |
| `cohorts_to_analyze` | Cancer types to evaluate | `['Bladder', 'Ovarian', 'Pancreatic']` |

## Output Files
1. **FeatureVectorTissueDet[1-N].csv** - Topological features for each biomarker combination
2. **Average1.xlsx** - Average distance metrics for all combinations
3. **Positives[1-40].xlsx** - Detection results for each offset threshold
4. **Top_10_Metrics_[1-40].xlsx** - Performance metrics for each offset
5. **Top_10_F1_ScoresBased_LocPerformance.xlsx** - Final ranked results

## Key Features
- **Calibration-Focused**: Optimized specifically for local calibration sample performance
- **Combinatorial Analysis**: Tests all possible 3-biomarker combinations
- **Threshold Optimization**: Tests 40 different offset thresholds
- **Multi-Metric Evaluation**: Uses F1 Score, Accuracy, Precision, Sensitivity
- **Tissue-Specific Optimization**: Finds best biomarkers for each cancer type

## Performance Metrics (Calibration Sample)
- **F1 Score**: Balance between precision and sensitivity
- **Accuracy**: Overall correct classification rate
- **Precision**: True positives among all positives
- **Sensitivity**: True positive detection rate

## Usage
1. Provide clinical data file path and sheet name
2. Specify biomarkers to test combinations on calibration sample
3. Set output directory for results
4. Define patient counts for each cancer type in calibration sample
5. Specify number of biomarker combinations to test
6. Run TDAcancer_detect_2_LC() function

## Significance
This calibration analysis provides:
- **Performance Baseline**: Establishes expected performance on calibration data
- **Biomarker Selection**: Identifies most effective 3-biomarker sets for each cancer type
- **Threshold Optimization**: Determines optimal parameters for each tissue
- **Validation Ready**: Prepares significant biomarker combinations for full dataset validation

The results from this local calibration analysis guide the selection of biomarker combinations for subsequent validation on the complete dataset.
-------------------------------------------------------------------------------------------------------------------------------------------------

# Tissue of Origin Localization Module - Full Dataset Validation

## Purpose
This module performs **validation of tissue localization performance** on the **entire dataset** (local calibration sample + test samples). It validates the biomarker combinations and threshold offsets identified during the local calibration phase, ensuring they maintain performance across the complete dataset.

## Overview
This validation step is crucial to:
- Confirm that biomarker combinations identified in calibration work on full data
- Verify threshold offsets remain effective with additional patients
- Ensure tissue localization performance generalizes to unseen data
- Validate the robustness of the calibration findings

## Main Function
**TDAcancer_detect_2()** - Executes the complete tissue localization validation pipeline on the full dataset

## Key Differences from Calibration Version
- **Uses entire dataset** instead of just calibration sample
- **Validates** rather than discovers optimal parameters
- **Tests all patients** for each cohort (not limited by sample size)
- **Confirms generalization** of calibration findings

## How It Works

### Part 1: Full Dataset Processing
- Processes clinical data with the same biomarker combinations identified during calibration
- Performs topological data analysis (TDA) on the complete dataset
- Calculates average topological features using all available patients
- Maintains consistency with calibration methodology

### Part 2: Validation with Identified Parameters
- Applies the same 40 threshold offsets (-20 to +19) used in calibration
- Uses biomarker combinations identified as optimal during calibration
- Tests performance across the entire patient population
- Validates that calibration results generalize to full data

### Part 3: Performance Validation
- Calculates performance metrics (Sensitivity, Precision, F1 Score, Accuracy) on full data
- Compares results with calibration performance
- Identifies any performance degradation or improvement
- Generates comprehensive validation reports

## Input Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `file_path` | Clinical Excel file path | `D:\Cancer Detection folder\...` |
| `sheet_name` | Excel sheet name | `'Normal and Cancer'` |
| `biomarkers` | Biomarkers identified during calibration | `['sc-kit/SCFR', 'CA-125', ...]` |
| `output_dir` | Output directory | `D:\Cancer Detection folder\Test 3` |
| `tumor_types` | Patient counts by type in full dataset | `{'normal': 47, 'bladder': 9, ...}` |
| `total_positive_patients` | **All** cancer patients by type | `{"Bladder": 26, "Ovarian": 32, ...}` |
| `num_combinations` | Number of biomarker combinations to validate | 56 |
| `random_patient_range` | **All** patients in full dataset | 322 |
| `cohorts_to_analyze` | Cancer types to validate | `['Bladder', 'Ovarian', 'Pancreatic']` |

## Output Files
1. **FeatureVectorTissueDet[1-N].csv** - Topological features for full dataset
2. **Average1.xlsx** - Average distance metrics for all combinations
3. **Positives[1-40].xlsx** - Validation results for each offset threshold
4. **Top_10_Metrics_[1-40].xlsx** - Performance metrics for each offset
5. **Top_10_F1_ScoresBased_LocPerformance.xlsx** - Final validation results

## Key Features
- **Validation-Focused**: Confirms calibration findings on full data
- **Consistent Methodology**: Uses same approaches as calibration phase
- **Comprehensive Testing**: Validates all identified biomarker combinations
- **Performance Comparison**: Allows direct calibration vs validation comparison
- **Generalization Assessment**: Tests robustness across complete dataset

## Performance Metrics (Validation)
- **F1 Score**: Balance between precision and sensitivity on full data
- **Accuracy**: Overall correct classification rate across all patients
- **Precision**: True positives among all positives in validation set
- **Sensitivity**: True positive detection rate on complete dataset

## Usage
1. Provide clinical data file path and sheet name for full dataset
2. Specify biomarker combinations identified during calibration
3. Set output directory for validation results
4. Define **complete** patient counts for each cancer type
5. Specify number of biomarker combinations to validate
6. Run TDAcancer_detect_2() function for validation

## Validation Significance
This step ensures that:
- **Calibration Findings Hold**: Parameters work beyond the calibration sample
- **Model Generalizes**: Performance maintains with additional patients
- **Clinical Applicability**: Results are reliable for real-world application
- **Robustness Confirmed**: Biomarker combinations are truly optimal

The validation results provide confidence that the tissue localization approach will perform reliably in clinical applications.
----------------------------------------------------------------------------------------------------------------------------------------------