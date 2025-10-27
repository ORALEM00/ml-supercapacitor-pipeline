import os 
import pandas as pd


def prepare_data():
    """ 
    Process data to be ready for ML models in data/processed.csv
    """
    input_path = "data/raw.csv" # File path for raw data
    output_path = "data/processed.csv" # File path to store processed data

    index_cols = "Num_Data" 
    df = pd.read_csv(input_path, index_col = index_cols)

    df = df.drop(["DOI (Reference)"], axis=1) # Drop reference column

    # Rename columns to appropiate names
    df.rename(columns={"Current_Density (A/g)": "Current_Density", 
                       "Specific_Capacitance (Fg-1)": "Specific_Capacitance",
                       "Material (M)": "Material",
                       "E_Concentration (M)": "E_Concentration",
                       "M_Density(g/cm3)": "M_Density",
                       "Current_Collector (CC)": "Current_Collector",
                       "Current_Collector (CC)": "Current_Collector",
                       "E_H_Cation_Radius" : "E_Cation_Radius",
                       "E_H_Anion_Radius" : "E_Anion_Radius",
                       }, inplace = True)
    
    # Remain only rows with writen input
    df.drop(df.tail(len(df["Electrode_ID"]) - 185).index, inplace = True)

    # Change to appropiate data type
    df['Electrode_ID'] = df['Electrode_ID'].astype('int64')
    df['Material'] = df['Material'].astype('int64')
    df['Is_Binder'] = df['Is_Binder'].astype('int64')
    df['Current_Collector'] = df['Current_Collector'].astype('int64')
    df['Morphology_Encoded'] = df['Morphology_Encoded'].astype('int64')
    df['Electrolyte_Type'] = df['Electrolyte_Type'].astype('int64')
    df['Binder_Type'] = df['Binder_Type'].astype('int64')
    df['Synthesis_Method'] = df['Synthesis_Method'].astype('int64')

    ############### CHECK THIS IF REMAIN #######################3
    df = df.drop(["Electrolyte_Type", "E_Anion_Conductivity", "E_Cation_Conductivity","E_Concentration"], axis = 1)

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    df.to_csv(output_path)
    print(f"Processed data saved to {output_path}")

    return
    

if __name__ == "__main__":
    prepare_data()