from typing import Dict, Tuple, List, Optional
import csv
import os
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
#from .ui_utils import style_dataframe
from .constants import ROWDICT as DEFAULT_ROWDICT
from pkg_resources import resource_string


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def create_iDot_worklist(data_input: Dict[str, pd.DataFrame], clean_labels: bool = False) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Generate iDot worklist from input plate data.
    
    Args:
        data_input (Dict[str, pd.DataFrame]): Dictionary containing source_id, source_vol, 
                                            target_id and target_vol DataFrames
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, int]]: (Worklist DataFrame, Count of failed transfers)
        
    Raises:
        ValueError: If volume or well requirements cannot be met
    """
    data_dict = {key: df.copy() for key, df in data_input.items()}
    worklist_entries = []
    
    for _, row in data_dict['target_vol'].iterrows():
        target_well = row['Well']
        liquid_name = data_dict['target_id'].loc[
            data_dict['target_id']['Well'] == target_well, 'Value'
        ].iloc[0]
        
        source_wells = data_dict['source_id'].loc[
            data_dict['source_id']['Value'] == liquid_name, 'Well'
        ]
        source_wells_df = data_dict['source_vol'][
            data_dict['source_vol']['Well'].isin(source_wells)
        ]
        
        valid_wells = validate_volumes(source_wells_df, row['Value'], liquid_name)
        source_well = valid_wells.iloc[0]
        
        data_dict['source_vol'].loc[
            data_dict['source_vol']['Well'] == source_well, 'Value'
        ] -= row['Value']
        
        worklist_entries.append({
            'Source Well': source_well,
            'Target Well': row['Target Well'],
            'Volume [uL]': row['Value'],
            'Liquid Name': liquid_name,
            'Additional Volume Per Source Well': 0
        })
    
    worklist_df = pd.DataFrame(worklist_entries).sort_values(by='Target Well')
    na_counts = worklist_df[worklist_df['Source Well'].isna()]['Liquid Name'].value_counts()
    
    if clean_labels:
        worklist_df['Liquid Name'] = worklist_df['Liquid Name'].str.replace(r'-\d+$', '', regex=True)
    
    return worklist_df.dropna(subset=['Source Well']), na_counts.to_dict()


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


# def process_file(file_obj: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
#     """
#     Process input file and generate HTML representations.
#     
#     Args:
#         file_obj (Path): Path to input Excel file
#         
#     Returns:
#         Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]: 
#             HTML strings for source_id, source_vol, target_id, target_vol tables and status message
#     """
#     if not file_obj:
#         return None, None, None, None, "Please provide both folder and file"
#     
#     file_path = file_obj
#     try:
# 
# 
#         dataframes = read_excel_sheets(file_path)
#         styled_dfs=style_dataframe(data_dict=dataframes)
# 
#         return (
#             styled_dfs['source_id'].to_html(),
#             styled_dfs['source_vol'].to_html(),
#             styled_dfs['target_id'].to_html(),
#             styled_dfs['target_vol'].to_html(),
#             styled_dfs['legend'].to_html()
# 
#         )
#     except Exception as e:
#         return None, None, None, None, f"Error: {str(e)}"

def generate_worklist(input_file: Path, output_folder: Path, plate_size: int = 1536, clean_labels: bool = False, header_config: dict = None) -> str:

    """
    Generate iDot worklist files from input data.
    
    Args:
        input_file (Path): Path to input Excel file containing plate data
        output_folder (Path): Directory where output files will be saved
        plate_size (int): Microplate format (96, 384 or 1536 wells)
        header_config (dict): Dictionary containing header configuration parameters
        
    Returns:
        str: Success message with generated output filenames
        
    Raises:
        ValueError: If plate_size is not 96, 384 or 1536
        FileNotFoundError: If input_file or output_folder don't exist
    """

    if not input_file or not output_folder:
        return "Please select input file and output folder"
    if not Path(input_file.name).exists():
        return f"Input file not found: {input_file.name}"
    
    if not Path(output_folder).exists():
        return f"Output folder not found: {output_folder}"

    # Read and process dataframes
    dataframes = read_excel_sheets(input_file.name)
    
    # Melt dataframes before processing
    if plate_size == 1536:
        ROWDICT = DEFAULT_ROWDICT
    elif plate_size in [384, 96]:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ROWDICT = dict(zip(letters, letters))
    else:
        raise ValueError("Invalid plate size. Please select 96, 384, or 1536.")
    melted_dataframes = {
        sheet_name: melt_and_combine(df, ROWDICT) 
        for sheet_name, df in dataframes.items()
    }
    
    # Generate worklist
    idot_wl, na_count = create_iDot_worklist(melted_dataframes, clean_labels=clean_labels)
    
    # Save outputs
    base_name = os.path.splitext(os.path.basename(input_file.name))[0]
    excel_output = os.path.join(output_folder, f"idot_worklist_{base_name}.xlsx")
    csv_output = os.path.join(output_folder, f"idot_worklist_{base_name}.csv")
    
    # Headers have 8 cols, thus add empty cols for consistency
    main_columns = idot_wl.columns.tolist()
    empty_cols = [''] * 3  
    csv_headers = main_columns + empty_cols
    for i in range(3):
        idot_wl[f'__empty_{i}'] = ''
    
    idot_wl.to_excel(excel_output, index=False, header=csv_headers)
    idot_wl.to_csv(csv_output, index=False, header=csv_headers)
    
    # Add headers to both files
    add_headers(excel_output, csv_output, **header_config)
    
    return f"Worklist generated: {os.path.basename(excel_output)} and {os.path.basename(csv_output)}"


def read_instructions() -> str:
    """
    Read markdown instructions file.
    
    Returns:
        str: Instructions text in markdown format
    """
    import os
    
    instructions = resource_string(__name__, 'resources/instructions.md')
    return instructions.decode('utf-8')

