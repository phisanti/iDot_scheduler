from typing import Dict, Tuple, List, Optional
import csv
import os
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from .constants import ROWDICT as DEFAULT_ROWDICT
from pkg_resources import resource_string

# Configuration
CONFIG = {
    'DEAD_VOLUME': 1,
    'MAX_SHEETS': 4,
    'VALID_PLATE_SIZES': {96, 384, 1536}
}

def read_excel_sheets(file_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Read and parse Excel sheets containing plate data.
    
    Args:
        file_path (Path): Path to the Excel file
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping sheet names to DataFrames
        
    Raises:
        FileNotFoundError: If file does not exist
        pd.errors.EmptyDataError: If file contains no data
    """
    try:
        excel_file = pd.ExcelFile(file_path)
        dataframes = {}
        
        for sheet in excel_file.sheet_names[:CONFIG['MAX_SHEETS']]:
            df = pd.read_excel(
                excel_file,
                sheet_name=sheet,
                skiprows=1,
                header=0,
                index_col=None
            )
            # Convert all numeric columns to float
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[numeric_cols] = df[numeric_cols].astype(float)
            dataframes[sheet] = df
            
        return dataframes
        
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        raise


def melt_and_combine(df: pd.DataFrame, row_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Transform plate layout DataFrame into melted format with well mappings.
    
    Args:
        df (pd.DataFrame): Input DataFrame with plate layout
        row_mapping (Dict[str, str]): Mapping of row labels to iDot format
        
    Returns:
        pd.DataFrame: Melted DataFrame with columns [Well, Row, Column, iDot_row, Target Well, Value]
    """
    if df.index.name and df.index.name.lower() in ['row', 'rows']:
        df = df.reset_index().rename(columns={df.index.name: 'Row'})
    
    melted_df = pd.melt(df, id_vars=['Row'], var_name='Column', value_name='Value')
    melted_df = melted_df.dropna(subset=['Value'])
    
    melted_df['Well'] = melted_df['Row'] + melted_df['Column'].astype(str)
    melted_df['iDot_row'] = melted_df['Row'].map(row_mapping)
    melted_df['Target Well'] = melted_df['iDot_row'] + melted_df['Column'].astype(str)
    
    return melted_df[['Well', 'Row', 'Column', 'iDot_row', 'Target Well', 'Value']]


def assign_channel_numbers(row_letter: str, plate_type: int, start_row: str = 'A', is_source: bool = True) -> int:
    """
    Assign channel numbers for parallel pipetting based on row letter and plate type.
    
    Args:
        row_letter (str): The row letter (A-Z)
        plate_type (str): Type of plate ('96', '384', or '1536')
        start_row (str): Starting row letter (default 'A')
        is_source (bool): Whether this is a source plate (default True)
    
    Returns:
        int: Channel number (1-4)
    """
    
    plate_type = int(plate_type)
    # For 96-well plates, always return 1
    if len(row_letter) == 1:
        # Single letter case (A-Z)
        row_index = ord(row_letter.upper()) - ord(start_row)
    else:
        # ZA-ZZ case
        base_index = 26  # Z is 26th letter
        second_letter = row_letter[1].upper()
        additional_index = ord(second_letter) - ord(start_row)
        row_index = base_index + additional_index
    
    # Single parallel set for 96-well plates
    if plate_type == 96:
        return 1

    # For 384-well plates, alternate between 1 and 2
    if plate_type == 384:
        return (row_index % 2) + 1
    
    # For 1536-well plates, cycle through 1-4
    if plate_type == 1536:
        return (row_index % 4) + 1
        
    return 0


def simplify_input_data(melted_data_frames, plate_type: str = '96') -> Dict[str, pd.DataFrame]:
    """
    Add parallel channel information and merge source/target data.
    
    Args:
        melted_data_frames (dict): Dictionary of DataFrames containing plate data
        plate_type (str): Type of target plate ('96', '384', or '1536')
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with merged source and target data
    """
    # Find minimum row for each dataframe separately
    min_rows = {
        key: df['Row'].min() 
        for key, df in melted_data_frames.items()
    }
    
    # Assign channel numbers to source and target plates
    for key, df in melted_data_frames.items():
        current_plate_type = '96' if key in ['source_id', 'source_vol'] else plate_type
        df['parallel_channel'] = df.apply(
            lambda row: assign_channel_numbers(
                row['Row'],
                current_plate_type,
                min_rows[key],
                key.startswith('source')
            ),
            axis=1
        )
    
    # Merge source and target data
    merged_source = melted_data_frames['source_id'].merge(
        melted_data_frames['source_vol'][['Well', 'Value']].rename(columns={'Value': 'vol'}),
        on='Well',
        how='left'
    )
    
    merged_target = melted_data_frames['target_id'].merge(
        melted_data_frames['target_vol'][['Well', 'Value']].rename(columns={'Value': 'vol'}),
        on='Well',
        how='left'
    )
    
    return {
        'source': merged_source,
        'target': merged_target
    }


def empty_plate(size: str, value_col_name: str) -> pd.DataFrame:
    """Create empty plate DataFrame with specified dimensions.
    Args: size (str) - Plate size ('96','384','1536'), value_col_name (str) - Column name for values
    Returns: Empty DataFrame with plate structure"""
    plate_dims = {
        '96': ('H', 12),
        '384': ('P', 24),
        '1536': ('ZF', 48)
    }
    
    last_row, num_cols = plate_dims[size]
    
    if size == '1536':
        row_labels = list(DEFAULT_ROWDICT.keys())
    else:
        row_labels = [chr(i) for i in range(ord('A'), ord(last_row) + 1)]
        
    col_labels = list(range(1, num_cols + 1))
    
    empty_df = pd.DataFrame(
        [(row, col, '') for row in row_labels for col in col_labels],
        columns=['idot_row', 'idot_col', value_col_name]
    )
    return empty_df

def validate_volumes(source_wells_df: pd.DataFrame, volume: float, liquid_name: str) -> pd.Series:
    """
    Validate volume requirements for liquid transfers.
    
    Args:
        source_wells_df (pd.DataFrame): DataFrame with source well information
        volume (float): Required transfer volume in μL
        liquid_name (str): Name of the liquid being transferred
        
    Returns:
        pd.Series: Valid source wells that meet volume requirements
        
    Raises:
        ValueError: If volume requirements cannot be met
    """
    dead_vol = CONFIG['DEAD_VOLUME']
    
    if source_wells_df.empty:
        raise ValueError(f"No source wells found for liquid {liquid_name}")
    
    wells_above_dead_vol = source_wells_df[source_wells_df['Value'] > dead_vol]
    if wells_above_dead_vol.empty:
        raise ValueError(f"No wells for {liquid_name} have more than the dead volume ({dead_vol}uL)")
    
    valid_wells = wells_above_dead_vol[wells_above_dead_vol['Value'] > volume + dead_vol]
    if valid_wells.empty:
        max_available = wells_above_dead_vol['Value'].max() - dead_vol
        raise ValueError(f"Requested volume ({volume}uL) for {liquid_name} exceeds maximum available ({max_available:.2f}uL)")
    
    return valid_wells['Well']


def add_headers(
    excel_path: str, 
    csv_path: str, 
    worklist_name: str,
    user: str,
    source_plate_type: str = "S.200 Plate",
    source_name: str = "Source Plate 1",
    target_type: str = "1536_Screenstar",
    target_name: str = "Target Plate 1",
    dispense_to_waste: bool = True,
    deionisation: bool = True,
    optimization: str = "ReorderAndParallel"
) -> None:
    """
    Add standardized headers to Excel and CSV worklist files.
    
    Args:
        excel_path (str): Path to Excel worklist file
        csv_path (str): Path to CSV worklist file  
        worklist_name (str): Name of the worklist
        user (str): Name of the user
        source_plate_type (str): Type of source plate
        source_name (str): Name of source plate
        target_type (str): Type of target plate
        target_name (str): Name of target plate
        dispense_to_waste (bool): Enable waste dispense
        deionisation (bool): Enable deionisation
        optimization (str): Optimization level
    """
    current_time = datetime.now()
    
    # Ensure 8 columns for each header row
    headers = [
        [worklist_name, '1.10.1.178', user, current_time.strftime("%d.%m.%Y"), 
         current_time.strftime("%H:%M"), '', '', ''],
        [source_plate_type, source_name, target_type, target_name, 'Waste Tube', '', '', ''],
        [f'DispenseToWaste={str(dispense_to_waste)}', 'DispenseToWasteCycles=3',
         'DispenseToWasteVolume=1e-7', f'UseDeionisation={str(deionisation)}',
         f'OptimizationLevel={optimization}', 'WasteErrorHandlingLevel=Ask',
         'SaveLiquids=Ask', '']
    ]

    # Update Excel
    df = pd.read_excel(excel_path)
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(headers).to_excel(writer, header=False, index=False)
        df.to_excel(writer, startrow=len(headers), index=False)
    
    # Update CSV with consistent column count
    with open(csv_path, 'r') as original:
        content = original.read()
    
    with open(csv_path, 'w', newline='') as modified:
        writer = csv.writer(modified)
        for row in headers:
            writer.writerow(row)
        modified.write(content)


def read_instructions() -> str:
    """
    Read markdown instructions file.
    
    Returns:
        str: Instructions text in markdown format
    """
    import os
    
    instructions = resource_string(__name__, 'resources/instructions.md')
    return instructions.decode('utf-8')

