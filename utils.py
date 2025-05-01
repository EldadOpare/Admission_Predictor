import pandas as pd

# Define valid unique values
unique_values_dict = {
    'gender': ['Female', 'Male'],
    'international': [False, True],
    'major': ['Business', 'Humanities', 'STEM'],
    'race': ['Asian', 'Black', 'International', 'Hispanic', 'White', 'Other'],
    'work_industry': [
        'Financial Services', 'Other', 'Technology', 'Consulting',
        'Nonprofit/Gov', 'PE/VC', 'Health Care', 'Investment Banking'
    ]
}

categorical_columns = list(unique_values_dict.keys())
columns_to_be_normalized = ['gpa', 'gmat', 'work_exp']


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:

    for col, unique_values in unique_values_dict.items():
        for val in unique_values:
            new_col_name = f"{col}{str(val).replace('/', '').replace(' ', '_')}"
            df[new_col_name] = (df[col] == val).astype(int)

    df.drop(columns=categorical_columns, inplace=True)

    for col in columns_to_be_normalized:
        std = df[col].std()
        if pd.isna(std) or std == 0:
            df[col] = 0 
        else:
            df[col] = (df[col] - df[col].mean()) / std

    return df

