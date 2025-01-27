import pandas as pd
import csv
from datetime import datetime

def read_excel_sheets(file_path):
    # Dictionary to store DataFrames
    dfs = {}
    
    # Read the Excel file
    excel_file = pd.ExcelFile(file_path)
    
    # Get the sheet names
    sheet_names = excel_file.sheet_names
    
    # Read each sheet
    for sheet in sheet_names[:4]:  
        df = pd.read_excel(
            excel_file,
            sheet_name=sheet,
            skiprows=1, 
            header=0,   
            index_col=None 
        )
        dfs[sheet] = df
    
    return dfs


def melt_and_combine(df, row_mapping):
    # Melt the DataFrame
    melted_df = pd.melt(df, id_vars=['Row'], var_name='Column', value_name='Value')
    
    melted_df = melted_df.dropna(subset=['Value'])
    melted_df['Well'] = melted_df['Row'] + melted_df['Column'].astype(str)
    melted_df['iDot_row'] = melted_df['Row'].map(row_mapping)
    melted_df['Target Well'] = melted_df['iDot_row'] + melted_df['Column'].astype(str)
    melted_df = melted_df[['Well', 'Row', 'Column', 'iDot_row', 'Target Well', 'Value']]
    
    return melted_df


#import copy
def create_iDot_worklist(data_input):
    data_dict = {key: df.copy() for key, df in data_input.items()}
    target_vol = data_dict['target_vol']
    target_id = data_dict['target_id']
    source_id = data_dict['source_id']
    source_vol = data_dict['source_vol']

    iDot_worklist = []

    for _, row in target_vol.iterrows():
        target_well = row['Well']
        target_iDot = row['Target Well']
        volume = row['Value']
        dead_vol = 1
        liquid_name = target_id.loc[target_id['Well'] == target_well, 'Value'].values[0]

        source_wells = source_id.loc[source_id['Value'] == liquid_name, 'Well']
        source_wells_df = source_vol[source_vol['Well'].isin(source_wells)]

        # Check for well location and volume
        if source_wells_df.empty:
            raise ValueError(f"No source wells found for liquid {liquid_name}")

        wells_above_dead_vol = source_wells_df[source_wells_df['Value'] > dead_vol]
        if wells_above_dead_vol.empty:
            raise ValueError(f"No wells for {liquid_name} have more than the dead volume ({dead_vol}uL)")
        wells_with_enough_vol = wells_above_dead_vol[wells_above_dead_vol['Value'] > volume + dead_vol]

        if wells_with_enough_vol.empty:
            max_available = wells_above_dead_vol['Value'].max() - dead_vol
            raise ValueError(f"Requested volume ({volume}uL) for {liquid_name} is higher than the maximum available volume ({max_available:.2f}uL)")

        valid_source_wells = source_vol.loc[
            (source_vol['Well'].isin(source_wells)) & 
            (source_vol['Value'] > dead_vol) &
            (source_vol['Value'] > volume), 'Well'
        ]

        source_well = valid_source_wells.iloc[0]
        remaining_volume = source_vol.loc[source_vol['Well'] == source_well, 'Value'].iloc[0]
        source_vol.loc[source_vol['Well'] == source_well, 'Value'] -= volume


        iDot_worklist.append({
            'Source Well': source_well,
            'Target Well': target_iDot,
            'Volume [uL]': volume,
            'Liquid Name': liquid_name,
            'Additional Volume Per Source Well' : 0
        })

        del source_well, source_wells_df, source_wells, wells_above_dead_vol, wells_with_enough_vol

    iDot_worklist_df=pd.DataFrame(iDot_worklist)
    iDot_worklist_df=iDot_worklist_df.sort_values(by='Target Well')
    na_counts = iDot_worklist_df[iDot_worklist_df['Source Well'].isna()]['Liquid Name'].value_counts().to_dict()
    df_filtered = iDot_worklist_df.dropna(subset=['Source Well']).sort_values(by='Source Well')

    return df_filtered, na_counts


def write_iDot_csv(df, filename, worklist_name, user):
    current_time = datetime.now()
    date_str = current_time.strftime("%d.%m.%Y")
    time_str = current_time.strftime("%H:%M")

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the first three special rows
        writer.writerow([worklist_name, '1.10.1.178', user, f"{date_str}", f"{time_str}"])
        writer.writerow(['S.100 Plate', 'Source Plate 1', '1536_Screenstar', 'Target Plate 1', 'Waste Tube'])
        writer.writerow(['DispenseToWaste=True', 'DispenseToWasteCycles=3', 'DispenseToWasteVolume=1e-7', 
                         'UseDeionisation=True', 'OptimizationLevel=ReorderAndParallel', 
                         'WasteErrorHandlingLevel=Ask', 'SaveLiquids=Ask'])
        
        # Write the DataFrame to CSV
        df.to_csv(csvfile, index=False, mode='a')

    print(f"CSV file '{filename}' has been created successfully.")
