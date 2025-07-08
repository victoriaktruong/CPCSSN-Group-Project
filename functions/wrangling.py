import pandas as pd

def load_csv(file_paths, delimiter='|'):
    """
    Load multiple CSV files from a dictionary of dataset names and file paths
    
    Parameters:
    - file_paths: dictionary of dataset names and file paths
    - delimiter: separator used in the CSV files
    
    Returns:
    - datasets: dictionary of loaded DataFrames with dataset names as keys
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
    - datasets: dictionary of DataFrames with only the selected columns
    """
    for name, columns in columns_to_keep.items():
        datasets[name] = datasets[name][columns]
    return datasets


def filter_by_values(df, column, values):
    """
    Filter rows where a column's value is within a list of allowed values
    
    Parameters:
    - df: DataFrame to filter
    - column: column name to filter on
    - values: list of allowed values to retain
    
    Returns:
    - Filtered DataFrame containing only allowed values
    """
    return df[df[column].isin(values)]


def replace_string_nan(df, column):
    """
    Replace string 'nan' with actual NaN values in the specified column

    Parameters:
    - df: DataFrame
    - column: column to clean

    Returns:
    - Updated DataFrame with 'nan' strings replaced by NaN
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
    - Cleaned DataFrame
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
    - lab_df: DataFrame containing lab results
    - diag_df: DataFrame containing diagnosis records
    - diagnosis_codes: list of diagnosis codes to identify BD patients
    - lab_test_names: list of lab test names to retain
    - patient_col: column name for patient identifier
    - lab_date_col: column name for lab test date
    - diag_code_col: column name for diagnosis code
    - diag_date_col: column name for diagnosis date
    - lab_name_col: column name for lab test name
    
    Returns:
    - DataFrame of lab tests occurring after BD diagnosis, with lab timing labeled
    """
    diag_df = diag_df[[patient_col, diag_code_col, diag_date_col]].copy()
    diag_df[diag_date_col] = pd.to_datetime(diag_df[diag_date_col], errors='coerce')
    first_dx = diag_df.sort_values([patient_col, diag_date_col]).groupby(patient_col).first().reset_index()
    first_dx = first_dx[first_dx[diag_code_col].isin(diagnosis_codes)]

    lab_df = lab_df[lab_df[lab_name_col].isin(lab_test_names)].copy()
    lab_df[lab_date_col] = pd.to_datetime(lab_df[lab_date_col], errors='coerce')

    merged = lab_df.merge(first_dx[[patient_col, diag_date_col]], on=patient_col, how='inner')
    merged['Lab_Timing'] = 'After'
    merged.loc[merged[lab_date_col] < merged[diag_date_col], 'Lab_Timing'] = 'Before'
    return merged[merged['Lab_Timing'] == 'After']


def classify_lab_timing(lab_df, diag_df, diagnosis_codes, lab_date_col='PerformedDate', diag_date_col='DateCreated'):
    """
    Classify patients based on whether lab tests occurred before, after, or both before and after the BD diagnosis
    
    Parameters:
    - lab_df: DataFrame of lab results
    - diag_df: DataFrame of diagnoses
    - diagnosis_codes: list of BD diagnosis codes
    - lab_date_col: name of the column with lab dates
    - diag_date_col: name of the column with diagnosis dates
    
    Returns:
    - summary: DataFrame showing each patient's lab data timing 
    - bd_first_clean: DataFrame of patients whose first diagnosis was a BD diagnosis
    """
    diag_df = diag_df.copy()
    diag_df[diag_date_col] = pd.to_datetime(diag_df[diag_date_col], errors='coerce')
    first_any_dx = diag_df.sort_values(['Patient_ID', diag_date_col]).groupby('Patient_ID').first().reset_index()
    first_any_dx = first_any_dx.rename(columns={diag_date_col: 'First_Diagnosis_Date',
                                                 'DiagnosisCode_calc': 'first_any_dx_code'})
    bd_first_clean = first_any_dx[first_any_dx['first_any_dx_code'].isin(diagnosis_codes)].copy()

    lab_df = lab_df.copy()
    lab_df[lab_date_col] = pd.to_datetime(lab_df[lab_date_col], errors='coerce')
    merged = lab_df.merge(bd_first_clean[['Patient_ID', 'First_Diagnosis_Date']], on='Patient_ID', how='inner')
    merged['Lab_Timing'] = 'After'
    merged.loc[merged[lab_date_col] < merged['First_Diagnosis_Date'], 'Lab_Timing'] = 'Before'

    summary = merged.groupby('Patient_ID')['Lab_Timing'].unique().reset_index()

    def classify(timings):
        if "Before" in timings and "After" in timings:
            return "Both before and after"
        elif "Before" in timings:
            return "Only before"
        elif "After" in timings:
            return "Only after"
        else:
            return "No lab data"

    summary['Lab_Data_Timing'] = summary['Lab_Timing'].apply(classify)
    return summary, bd_first_clean


def other_dx_same_day(diag_df, bd_first_clean, diag_date_col='DateCreated'):
    """
    Count patients who had a non-BD diagnosis recorded on the same day as their first BD diagnosis
    
    Parameters:
    - diag_df: DataFrame of all diagnosis entries
    - bd_first_clean: DataFrame of patients whose first diagnosis was BD
    - diag_date_col: name of the column with diagnosis dates
    
    Returns:
    - Number of patients with non-BD diagnoses on the same day as their BD diagnosis
    """
    merged = diag_df.merge(bd_first_clean[['Patient_ID', 'First_Diagnosis_Date']], on='Patient_ID', how='inner')
    same_day = merged[merged[diag_date_col] == merged['First_Diagnosis_Date']]
    non_bd = same_day[~same_day['DiagnosisCode_calc'].isin(bd_codes)]
    return non_bd['Patient_ID'].nunique()


