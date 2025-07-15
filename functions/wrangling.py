import pandas as pd

def load_csv(file_paths, delimiter='|'):
    """
    Load multiple CSV files from a dictionary of dataset names and file paths
    
    Parameters:
    - file_paths: dictionary of dataset names and file paths
    - delimiter: separator used in the CSV files
    
    Returns:
    - dictionary of loaded DataFrames with dataset names as keys
    """
    datasets = {}
    for name, path in file_paths.items():
        datasets[name] = pd.read_csv(path, sep=delimiter, on_bad_lines='skip', low_memory=False)
    return datasets

def select_columns(datasets, columns_to_keep):
    """
    Select specific columns for each dataset based on a provided dictionary
    
    Parameters:
    - datasets: dictionary of DataFrames with dataset names as keys
    - columns_to_keep: dictionary mapping dataset names to a list of columns to keep
    
    Returns:
    - dictionary of DataFrames with only the selected columns
    """
    for name, columns in columns_to_keep.items():
        datasets[name] = datasets[name][columns]
    return datasets


def replace_string_nan(df, column):
    """
    Replace string 'nan' with actual NaN values in the specified column

    Parameters:
    - df: DataFrame
    - column: column to clean

    Returns:
    - updated DataFrame with 'nan' strings replaced by NaN
    """
    df.loc[df[column].str.lower() == "nan", column] = pd.NA
    return df


def preprocess_data(df, cleaning_steps):
    """
    Apply multiple cleaning steps to a DataFrame
    
    Parameters:
    - df: DataFrame to clean
    - cleaning_steps: list of tuples (column_name, method)
        - Method:
            - uppercase: converts strings to uppercase
            - strip: removes leading or trailing spaces
            - dropna: drops rows where column is missing
            - datetime: converts to datetime format
    
    Returns:
    - cleaned DataFrame
    """
    for col, method in cleaning_steps:
        if method == 'uppercase':
            df[col] = df[col].fillna('').str.upper()
        elif method == 'strip':
            df[col] = df[col].astype(str).str.strip()
        elif method == 'dropna':
            df = df.dropna(subset=[col])
        elif method == 'datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def get_first_diagnosis(df, diagnosis_col, date_col, patient_col='Patient_ID'):
    """
    Get the first diagnosis record per patient based on the diagnosis date

    Parameters:
    - df: DataFrame with diagnosis records
    - diagnosis_col: column name for the diagnosis code or text
    - date_col: column name for the diagnosis date
    - patient_col: column name for patient ID

    Returns:
    - DataFrame with the first diagnosis per patient
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df.sort_values([patient_col, date_col]).groupby(patient_col).first().reset_index()


def get_first_matching_diagnosis(df, diagnosis_col, date_col, target_codes, patient_col='Patient_ID', 
                                 new_date_col='Diagnosis_Date',
                                 new_code_col='Diagnosis_Code'):
    """
    Find the first diagnosis by date for each patient that matches the diagnosis codes

    Parameters:
    - df: DataFrame with diagnosis records
    - diagnosis_col: column name for diagnosis code
    - date_col: column name for diagnosis date
    - target_codes: list of codes to filter on
    - patient_col: column name for patient ID
    - new_date_col: name for renamed diagnosis date column
    - new_code_col: name for renamed diagnosis code column

    Returns:
    - DataFrame with one row per patient for first matching diagnosis
    """
    first_dx = get_first_diagnosis(df, diagnosis_col, date_col, patient_col)
    first_dx = first_dx[first_dx[diagnosis_col].isin(target_codes)].copy()
    return first_dx.rename(columns={date_col: new_date_col, diagnosis_col: new_code_col})


def extract_labs_relative_to_diagnosis(
    lab_df,
    diag_df,
    diagnosis_codes,
    lab_test_names,
    patient_col='Patient_ID',
    lab_date_col='PerformedDate',
    diag_code_col='DiagnosisCode_calc',
    diag_date_col='DateCreated',
    lab_name_col='Name_calc',
):
    """
    Extract lab tests that occur after the first bipolar disorder diagnosis
    
    Parameters:
    - lab_df: DataFrame with lab results
    - diag_df: DataFrame with diagnosis records
    - diagnosis_codes: list of diagnosis codes to identify BD patients
    - lab_test_names: list of lab test names to keep
    - patient_col: column name for patient ID
    - lab_date_col: column name for lab test date
    - diag_code_col: column name for diagnosis code
    - diag_date_col: column name for diagnosis date
    - lab_name_col: column name for lab test name
    
    Returns:
    - DataFrame of lab tests occurring after BD diagnosis, with lab timing labeled
    """
    diag_df = diag_df[[patient_col, diag_code_col, diag_date_col]].copy()
    diag_df[diag_date_col] = pd.to_datetime(diag_df[diag_date_col], errors='coerce')
    first_dx = get_first_diagnosis(diag_df, diag_code_col, diag_date_col)
    first_dx = first_dx[first_dx[diag_code_col].isin(diagnosis_codes)]

    lab_df = lab_df[lab_df[lab_name_col].isin(lab_test_names)].copy()
    lab_df[lab_date_col] = pd.to_datetime(lab_df[lab_date_col], errors='coerce')

    merged = lab_df.merge(first_dx[[patient_col, diag_date_col]], on=patient_col, how='inner')
    merged['Lab_Timing'] = 'After'
    merged.loc[merged[lab_date_col] < merged[diag_date_col], 'Lab_Timing'] = 'Before'
    return merged[merged['Lab_Timing'] == 'After']


def classify_lab_timing(
    lab_df,
    diag_df,
    diagnosis_codes,
    relevant_markers=None,
    lab_date_col='PerformedDate',
    diag_date_col='DateCreated',
    lab_name_col='Name_calc',
    diag_code_col='DiagnosisCode_calc'
):
    """
    Classify patients based on whether lab tests occurred before, after,
    or both before and after their first BD diagnosis

    Parameters:
    - lab_df: DataFrame with lab records
    - diag_df: DataFrame with diagnosis records
    - diagnosis_codes: list of BD ICD codes to filter on
    - relevant_markers: list of lab test names to include
    - lab_date_col: column name for lab test date
    - diag_date_col: column name for diagnosis date
    - lab_name_col: column name for lab test name
    - diag_code_col: column name for diagnosis code

    Returns:
    - summary: DataFrame with one row per patient, showing lab timing classification
    - bd_first_clean: DataFrame of patients whose first-ever diagnosis was BD
    """
    # Get first-ever diagnosis per patient
    diag_df = diag_df.copy()
    diag_df[diag_date_col] = pd.to_datetime(diag_df[diag_date_col], errors='coerce')
    first_any_dx = diag_df.sort_values(['Patient_ID', diag_date_col]).groupby('Patient_ID').first().reset_index()
    first_any_dx = first_any_dx.rename(columns={diag_date_col: 'First_Diagnosis_Date',
                                                diag_code_col: 'first_any_dx_code'})

    # Keep only patients whose first diagnosis is BD
    bd_first_clean = first_any_dx[first_any_dx['first_any_dx_code'].isin(diagnosis_codes)].copy()

    # Filter labs by relevant markers if provided
    lab_df = lab_df.copy()
    if relevant_markers is not None:
        lab_df = lab_df[lab_df[lab_name_col].isin(relevant_markers)]

    # Process lab dates and merge with diagnosis
    lab_df[lab_date_col] = pd.to_datetime(lab_df[lab_date_col], errors='coerce')
    merged = lab_df.merge(bd_first_clean[['Patient_ID', 'First_Diagnosis_Date']], on='Patient_ID', how='inner')

    # Label each lab result by timing
    merged['Lab_Timing'] = merged.apply(
        lambda row: 'Before' if row[lab_date_col] < row['First_Diagnosis_Date']
        else 'After' if row[lab_date_col] > row['First_Diagnosis_Date']
        else 'Same day',
        axis=1
    )
    # Classify lab timing per patient
    def classify_patient(timings):
        timings = set(timings)
        if 'Before' in timings and 'After' in timings:
            return 'Both'
        elif 'After' in timings:
            return 'Only after'
        elif 'Before' in timings:
            return 'Only before'
        else:
            return 'No lab data'

    summary = merged.groupby('Patient_ID')['Lab_Timing'].apply(classify_patient).reset_index()
    summary.columns = ['Patient_ID', 'Lab_Data_Timing']

    return summary, bd_first_clean


def pivot_lab_data(df, index_cols, name_col, value_col):
    """
    Pivot lab data to create one row per patient per test date, with lab test names as columns.

    Parameters:
    - df: DataFrame of lab results
    - index_cols: list of column names to use as the row index 
    - name_col: column name for lab test names 
    - value_col: column name for lab test results 

    Returns:
    - pivoted DataFrame with lab test names as columns and lab test values
    """
    return df.pivot_table(index=index_cols, columns=name_col, values=value_col, aggfunc='first').reset_index()


def add_demographics_to_labs(
    lab_df,patient_df,
    id_col='Patient_ID',
    sex_col='Sex',
    birth_col='BirthYear',
    lab_date_col='PerformedDate'
):
    """
    Add patient sex and birth year into lab results and calculate age at time of lab

    Returns:
        DataFrame with 'Sex' and 'Age' columns added
    """
    patient_df = patient_df[[id_col, sex_col, birth_col]].copy()
    lab_df = lab_df.merge(patient_df, on=id_col, how='left')
    lab_df['Age'] = lab_df[lab_date_col].dt.year - lab_df[birth_col]
    return lab_df


def split_patients_by_lab_timing(lab_timing_df, timing_col='Lab_Data_Timing', patient_col='Patient_ID'):
    """
    Separate patient IDs into groups based on lab timing classification

    Parameters:
    - lab_timing_df: DataFrame with lab timing summary
    - timing_col: column that contains timing labels
    - patient_col: column with patient IDs

    Returns:
    - dictionary with keys 'only_before', 'only_after', 'both', each mapping to Patient IDs
    """
    return {
        'only_before': lab_timing_df[lab_timing_df[timing_col] == 'Only before'][patient_col],
        'only_after': lab_timing_df[lab_timing_df[timing_col] == 'Only after'][patient_col],
        'both': lab_timing_df[lab_timing_df[timing_col] == 'Both'][patient_col]
    }


def other_dx_same_day(diag_df, bd_first_clean, bd_codes, diag_date_col='DateCreated'):
    """
    Count patients who had a non-BD diagnosis recorded on the same day as their first BD diagnosis
    
    Parameters:
    - diag_df: DataFrame of all diagnosis entries
    - bd_first_clean: DataFrame of patients whose first diagnosis was BD
    - diag_date_col: name of the column with diagnosis dates
    
    Returns:
    - number of patients with non-BD diagnoses on the same day as their BD diagnosis
    """
    merged = diag_df.merge(bd_first_clean[['Patient_ID', 'First_Diagnosis_Date']], on='Patient_ID', how='inner')
    same_day = merged[merged[diag_date_col] == merged['First_Diagnosis_Date']]
    non_bd = same_day[~same_day['DiagnosisCode_calc'].isin(bd_codes)]
    return non_bd['Patient_ID'].nunique()


def non_bd_same_day_with_labs_after(
    diag_df, bd_labs_after_df, 
    bd_first_clean, bd_codes,
    patient_col='Patient_ID',
    diag_code_col='DiagnosisCode_calc',
    diag_date_col='DateCreated'
):
    """
    Count how many patients with lab results after BD diagnosis also had a non-BD diagnosis
    on the same day as their BD diagnosis

    Parameters:
    - diag_df: DataFrame of all diagnoses
    - bd_labs_after_df: DataFrame of lab results after BD diagnosis
    - bd_first_clean: DataFrame of patients whose first diagnosis was BD
    - bd_codes: list of BD ICD-9 codes
    - patient_col: name of patient ID column
    - diag_code_col: name of diagnosis code column
    - diag_date_col: name of diagnosis date column

    Returns:
    - count of patients with non-BD diagnosis on same day as BD diagnosis and labs after
    """
    # Filter diagnosis dataframe to same-day non-BD diagnoses
    merged = diag_df.merge(bd_first_clean[[patient_col, 'First_Diagnosis_Date']], on=patient_col, how='inner')
    merged[diag_date_col] = pd.to_datetime(merged[diag_date_col], errors='coerce')
    same_day = merged[merged[diag_date_col] == merged['First_Diagnosis_Date']]
    non_bd_same_day = same_day[~same_day[diag_code_col].isin(bd_codes)]

    # Get patient IDs from bd_labs_after_df
    labs_after_patients = set(bd_labs_after_df[patient_col].unique())

    # Combine those with non-BD same-day diagnoses
    overlap = set(non_bd_same_day[patient_col].unique()) & labs_after_patients
    return len(overlap)


def print_summary_stats(stats):
    """
    Print formatted summary statistics from a list of (label, value) pairs

    Parameters:
    - stats: list of tuples with a string label and a value

    Returns:
    - prints output to console
    """
    for label, value in stats:
        print(f"{label}: {value}")
