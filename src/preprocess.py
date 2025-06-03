import numpy as np 
import pandas as pd 

def assign_pressure(data):
    # Step 1: Create masks for students and professionals
    is_student = data["Working Professional or Student"] == "Student"
    is_professional = data["Working Professional or Student"] == "Working Professional"

    # Step 2: Assign values using np.where()
    data["Pressure"] = np.where(
        is_student,
        data["Academic Pressure"],  # Use academic pressure for students
        np.where(
            is_professional,
            data["Work Pressure"],  # Use work pressure for professionals
            np.where(
                data["Academic Pressure"].notna() & data["Work Pressure"].notna(),
                data[["Academic Pressure", "Work Pressure"]].max(axis=1),  # Max for hybrids
                np.nan  # Fallback for edge cases
            )   
        )
    )

    return data 

def assign_satisfaction(data):
    # Step 1: Create masks for students and professionals
    is_student = data["Working Professional or Student"] == "Student"
    is_professional = data["Working Professional or Student"] == "Working Professional"

    # Step 2: Assign values using np.where()
    data["Satisfaction"] = np.where(
        is_student,
        data["Study Satisfaction"],  # Use study satisfaction for students
        np.where(
            is_professional,
            data["Job Satisfaction"],  # Use job satisfaction for professionals
            np.where(
                data["Study Satisfaction"].notna() & data["Job Satisfaction"].notna(),
                data[["Study Satisfaction", "Job Satisfaction"]].mean(axis=1),  # Mean for hybrids
                np.nan  # Fallback for edge cases
            )   
        )
    )

    return data 



def fill_profession(data): 
    data["Profession"] = np.where(
        data["Working Professional or Student"] == "Working Professional",
        data["Profession"],
        "Not Applicable"
    )

    return data 

def replace_diet_habits(data):
    top_categories = ["Moderate", "Unhealthy", "Healthy"]
    data['Dietary Habits'] = np.where(
        data['Dietary Habits'].isin(top_categories),
        data['Dietary Habits'],  # Keep if in top_categories
        'Other'                  # Else replace with "Other"
    )

    return data 

def replace_sleep_duration(data):
    top_categories = ["Less than 5 hours", "7-8 hours", "More than 8 hours", "5-6 hours"]
    data["Sleep Duration"] = np.where(
        data["Sleep Duration"].isin(top_categories),
        data["Sleep Duration"],  # Keep if in top_categories
        "other"                  # Else replace with "Other"
    )

    return data 