import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin

def assign_pressure(data):
    """
    Create unified 'Pressure' feature from academic/work pressure columns based on occupation.

    For students: Uses 'Academic Pressure'
    For professionals: Uses 'Work Pressure'
    For others: Uses maximum of available pressure values or NaN

    Args:
        data (pd.DataFrame):  
            Input dataframe containing:
            - 'Working Professional or Student' (str): Occupation category
            - 'Academic Pressure' (float): Student pressure (1-5 scale)
            - 'Work Pressure' (float): Professional pressure (1-5 scale)

    Returns:
        pd.DataFrame:
            Original DataFrame with new 'Pressure' column (float, 1-5 scale)
    """
    
    data=data.copy()
    is_student = data["Working Professional or Student"] == "Student"
    is_professional = data["Working Professional or Student"] == "Working Professional"

    data["Pressure"] = np.where(
        is_student,
        data["Academic Pressure"],
        np.where(
            is_professional,
            data["Work Pressure"], 
            np.where(
                data["Academic Pressure"].notna() & data["Work Pressure"].notna(),
                data[["Academic Pressure", "Work Pressure"]].max(axis=1),  # Max for hybrids
                np.nan  # Fallback for edge cases
            )   
        )
    )

    return data 

def assign_satisfaction(data):
    """
    Create unified 'Satisfaction' feature from study/job satisfaction columns.

    For students: Uses 'Study Satisfaction'
    For professionals: Uses 'Job Satisfaction'
    For others: Uses average of available satisfaction values or NaN

    Args:
        data (pd.DataFrame):  
            Input dataframe containing:
            - 'Working Professional or Student' (str): Occupation category
            - 'Academic Satisfaction' (float): Student satisfaction (1-5 scale)
            - 'Work Satisfaction' (float): Professional satisfaction (1-5 scale)

    Returns:
        pd.DataFrame:
            Original DataFrame with new 'Satisfaction' column (float, 1-5 scale)
    """

    data=data.copy()
    is_student = data["Working Professional or Student"] == "Student"
    is_professional = data["Working Professional or Student"] == "Working Professional"

    data["Satisfaction"] = np.where(
        is_student,
        data["Study Satisfaction"],  
        np.where(
            is_professional,
            data["Job Satisfaction"],  
            np.where(
                data["Study Satisfaction"].notna() & data["Job Satisfaction"].notna(),
                data[["Study Satisfaction", "Job Satisfaction"]].mean(axis=1),  # Mean for hybrids
                np.nan  # Fallback for edge cases
            )   
        )
    )

    return data 

def replace_diet_habits(data):
    """
    Standardize 'Dietary Habits' by grouping rare categories into 'Other'.

    Args:
        data (pd.DataFrame): 
            Input DataFrame containing 'Dietary Habits' column.

    Returns:
        pd.DataFrame:
            Original DataFrame with standardized dietary categories:
            - 'Healthy', 'Moderate', 'Unhealthy', or 'Other'
    """
     
    data=data.copy()
    top_categories = ["Moderate", "Unhealthy", "Healthy"]
    data['Dietary Habits'] = np.where(
        data['Dietary Habits'].isin(top_categories),
        data['Dietary Habits'],  
        'Other'                  
    )

    return data 

def replace_sleep_duration(data):
    """
    Standardize 'Sleep Duration' by grouping rare categories into 'other'

    Args:
        data (pd.DataFrame): 
            Input DataFrame containing 'Sleep Duration' column.

    Returns:
        pd.DataFrame:
            Original DataFrame with standardized sleep categories:
            - 'Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours', or 'other'
    """
     
    data=data.copy()
    top_categories = ["Less than 5 hours", "7-8 hours", "More than 8 hours", "5-6 hours"]
    data["Sleep Duration"] = np.where(
        data["Sleep Duration"].isin(top_categories),
        data["Sleep Duration"],  
        "other"                 
    )

    return data 

class GroupImputer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for group-wise median imputation.

    Imputes missing values using medians calculated separately for:
    - Students
    - Working Professionals

    Attributes:   
        cols_to_impute : list of str
            Column names to impute.

        medians_ : dict
            Dictionary storing learned medians in format:
            {column_name: {'Student': median, 'Working Professional': median}}
    """

    def __init__(self, cols_to_impute):
        """
        Initialize the imputer with columns to process.

        args:
            cols_to_impute : list of str
                Column names to impute.
        """

        self.cols_to_impute = cols_to_impute
        self.medians = {} # storage for learned values
    
    def fit(self, X):
        """
        Learn median values for each group (students/professionals).

        args:
            X (pd.DataFrame):
                Training data containing subset columns and 'Working Professional or Student'.

        Returns:
            self:
                Fitted imputer.
        """

        for col in self.cols_to_impute:
            self.medians[col] = X.groupby("Working Professional or Student")[col].median().to_dict()
        return self
    
    def transform(self, X):
        """
        Apply learned median imputation to the data.

        args:
            X (pd.DataFrame):
                Data to transform.

        Returns:
            pd.DataFrame:
                Transformed data with missing values imputed.
        """
        
        X = X.copy()
        for col in self.cols_to_impute: 
            missing_mask = X[col].isna()
            for group, median_val in self.medians[col].items():
                group_mask = (X["Working Professional or Student"] == group) & missing_mask
                X.loc[group_mask, col] = median_val
        return X
    
def add_pressure_ratio(X):
    """Add Pressure/Satisfaction ratio feature with epsilon to avoid division by zero
    
    args:
        X (pd.DataFrame):
            Data to transform.
    
    Returns:
        pd.DataFrame:
            Transformed data with Pressure/Satisfaction ratio
    """
    X = X.copy()
    X["Pressure_Satisfaction_Ratio"] = X["Pressure"] / (X["Satisfaction"] + 1e-6)
    return X