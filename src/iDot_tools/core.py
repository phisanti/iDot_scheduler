from typing import Dict, Tuple
import csv
import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from .utils import read_excel_sheets, melt_and_combine, simplify_input_data, CONFIG
from .constants import ROWDICT as DEFAULT_ROWDICT
from .constants import ParallelisationType

# Set up logger
logger = logging.getLogger('iDot_tools')
def setup_logging(output_folder: Path, base_name: str) -> None:
    global logger
    logger.setLevel(logging.DEBUG)
    output_folder = Path(output_folder)
    log_file = output_folder / f"idot_worklist_{base_name}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


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


def update_source_volume(data_dict: Dict[str, pd.DataFrame], 
                        source_well: str, 
                        transfer_vol: float) -> None:
    data_dict['source'].loc[data_dict['source']['Well'] == source_well, 'vol'] -= transfer_vol


def process_sequential_transfers(data_dict: Dict[str, pd.DataFrame], 
                               remaining_targets: pd.DataFrame,
                               dead_vol: float) -> list:
    worklist_entries = []
    
    for _, row in remaining_targets.iterrows():
        source_wells = data_dict['source'].loc[
            data_dict['source']['Value'] == row['Value']
        ]
        valid_source = source_wells[source_wells['vol'] >= (row['vol'] + dead_vol)]
        
        if not valid_source.empty:
            source_well = valid_source.iloc[0]['Well']
            update_source_volume(data_dict, source_well, row['vol'])
            worklist_entries.append(
                create_worklist_entry(source_well, row['Target Well'], row['vol'], row['Value'])
            )
    
    return worklist_entries


def find_matching_sequence_v0(source_df, target_sequence, target_vols, dead_vol=1):
    sequences = {}
    dead_vol = [dead_vol] * len(target_sequence)
    logger.debug(f"Finding matches for sequence of length {len(target_sequence)}")

    for col in source_df['Column'].unique():
        source_data = source_df[source_df['Column'] == col]
        col_matches = []  # Store matches for each window in this column
        
        for i in range(len(source_data) - len(target_sequence) + 1):
            id_window = source_data.iloc[i:i+len(target_sequence)]
            wells = id_window['Well'].tolist()
            
            match_count = sum((id_window['Value'] == target_sequence) & 
                            (id_window['vol'] >= target_vols) & 
                            (id_window['vol'] >= dead_vol))
            col_matches.append({'wells': wells, 'matches': match_count})
            logger.debug(f'Wells in window: {id_window['Value']}')
            logger.debug(f"Value matches: {(id_window['Value'] == target_sequence)}, vol matches: {(id_window['vol'] >= target_vols)}, dead vol matches: {(id_window['vol'] >= dead_vol)}")
            logger.debug(f"Column {col}, Window {i}: {match_count} matches")

        
        if col_matches:  # If we found any matches in this column
            # Store the window with maximum matches for this column
            best_window = max(col_matches, key=lambda x: x['matches'])
            sequences[col] = best_window

    if sequences:
        # Find column with highest matches (first one in case of tie)
        best_col = max(sequences.items(), key=lambda x: x[1]['matches'])[0]
        if sequences[best_col]['matches'] > 0:
            return sequences[best_col]['wells']
    return [], target_sequence


def find_matching_sequence(source_df, target_sequence, target_vols, dead_vol=1):
    sequences = {}
    dead_vol = [dead_vol] * len(target_sequence)
    logger.debug(f"Starting sequence search with target: {target_sequence}, vols: {target_vols}, dead_vol: {dead_vol}")
    
    for col in source_df['Column'].unique():
        logger.debug(f"Processing column {col}")
        source_data = source_df[source_df['Column'] == col]
        col_matches = []
        
        for i in range(len(source_data) - len(target_sequence) + 1):
            logger.debug(f"Checking window at position {i} in column {col}")
            id_window = source_data.iloc[i:i+len(target_sequence)]
            wells = id_window['Well'].tolist()
            
            match_mask = (id_window['Value'] == target_sequence) & \
                        (id_window['vol'] >= target_vols) & \
                        (id_window['vol'] >= dead_vol)
            
            match_count = sum(match_mask)
            logger.debug(f'Wells in window: {id_window['Value']}')
            logger.debug(f"Value matches: {(id_window['Value'] == target_sequence)}, vol matches: {(id_window['vol'] >= target_vols)}, dead vol matches: {(id_window['vol'] >= dead_vol)}")
            logger.debug(f"Window match count: {match_count}, match pattern: {match_mask.tolist()}")

            filtered_wells = [w for w, m in zip(wells, match_mask) if m]
            filtered_target = [t for t, m in zip(target_sequence, match_mask) if m]
            logger.debug(f"Filtered wells: {filtered_wells}, filtered target: {filtered_target}")
            
            col_matches.append({
                'wells': filtered_wells,
                'matches': match_count,
                'match_mask' : match_mask.tolist()
            })
        
        if col_matches:
            best_window = max(col_matches, key=lambda x: x['matches'])
            sequences[col] = best_window
            logger.debug(f"Best window for column {col}: matches={best_window['matches']}, wells={best_window['wells']}")

    if sequences:
        best_col = max(sequences.items(), key=lambda x: x[1]['matches'])[0]
        logger.debug(f"Selected best column {best_col} with {sequences[best_col]['matches']} matches")
        if sequences[best_col]['matches'] > 0:
            return sequences[best_col]['wells'], sequences[best_col]['match_mask']
    
    logger.debug("No matches found, returning empty lists")
    return [], []

def process_parallel_transfers(data_dict: Dict[str, pd.DataFrame], dead_vol: float) -> Tuple[list, set]:
    logger.info("Starting parallel transfer processing")
    logger.info(f"Initial target wells count: {len(data_dict['target'])}")

    worklist_entries = []
    processed_wells = set()
    
    max_parallel_channels = data_dict['target']['parallel_channel']
    unique_cols = data_dict['target']['Column'].unique()
    
    for channel_i in range(1, max_parallel_channels.max() + 1):
        logger.debug(f"Processing channel {channel_i}")
        for col_i in unique_cols:
            target_subset = data_dict['target'].loc[
                (data_dict['target']['Column'] == col_i) &
                (data_dict['target']['parallel_channel'] == channel_i)
            ]

            if len(target_subset) == 0:
                continue
                
            target_sequence = target_subset['Value'].tolist()
            target_vols = target_subset['vol'].tolist()
            matching_wells, match_mask = find_matching_sequence(
                data_dict['source'], 
                target_sequence, 
                target_vols, 
                dead_vol
            )

            logger.debug(f"Column {col_i}: Found {len(target_subset)} targets")
            if matching_wells:
                logger.debug(f"Matched sequence in column {col_i}: {matching_wells}")
                filtered_target_subset = target_subset[match_mask]
                for value, target_well, source_well, transfer_vol in zip(
                    filtered_target_subset['Value'], 
                    filtered_target_subset['Target Well'],
                    matching_wells, 
                    filtered_target_subset['vol']
                ):
                    update_source_volume(data_dict, source_well, transfer_vol)
                    processed_wells.add(target_well)
                    worklist_entries.append(
                        create_worklist_entry(source_well, target_well, transfer_vol, value)
                    )

    logger.info(f"Completed parallel transfers: {len(worklist_entries)}")
    return worklist_entries, processed_wells

def create_worklist_entry(source_well: str, 
                         target_well: str, 
                         transfer_vol: float, 
                         liquid_name: str) -> dict:
    return {
        'Source Well': source_well,
        'Target Well': target_well,
        'Volume [uL]': transfer_vol,
        'Liquid Name': liquid_name,
        'Additional Volume Per Source Well': 0
    }


def create_iDot_worklist(
    data_dict: Dict[str, pd.DataFrame], 
    dead_vol: float = 20.,
    use_parallel: bool = True,
    clean_labels: bool = False
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Create iDot worklist with configurable parallelisation and label cleaning.
    
    Args:
        data_input: Input data dictionary
        dead_vol: Dead volume in μL
        use_parallel: Whether to use parallel transfers
        clean_labels: Whether to clean numerical suffixes from labels
        
    Returns:
        Tuple of worklist DataFrame and count of failed transfers
    """
    logger.info("=== Creating Worklist ===")
    logger.debug(f"Initial data keys: {data_dict.keys()}")
    worklist_entries = []
    processed_wells = set()

    if use_parallel:
        # First pass - parallel channels
        logger.debug(f"Target data:\n{data_dict['target']}")
        logger.debug(f"Initial worklist entries count: {len(worklist_entries)}")
        logger.info(f"Unique parallel channels: {data_dict['target']['parallel_channel'].unique()}")

        worklist_entries, processed_wells = process_parallel_transfers(data_dict, dead_vol)
        parallel_df = pd.DataFrame(worklist_entries)
        if not parallel_df.empty:
            parallel_dupes = parallel_df[parallel_df['Target Well'].duplicated(keep=False)]
            if not parallel_dupes.empty:
                logger.warning("Duplicates found after parallel processing:")
                logger.warning(f"Duplicate wells:\n{parallel_dupes[['Source Well', 'Target Well', 'parallel_channel']]}")
   
    # Process remaining/all wells sequentially
    remaining_targets = data_dict['target'][~data_dict['target']['Target Well'].isin(processed_wells)]
    sequential_entries = process_sequential_transfers(data_dict, remaining_targets, dead_vol)
    worklist_entries.extend(sequential_entries)
    logger.info(f"Processing remaining targets: {len(remaining_targets)}")
    logger.info(f"Final worklist entries: {len(worklist_entries)}")

    worklist_df = pd.DataFrame(worklist_entries).sort_values(by='Target Well')
    
    if clean_labels:
        worklist_df['Liquid Name'] = worklist_df['Liquid Name'].str.replace(r'-\d+$', '', regex=True)
        logger.debug("Labels cleaned of numerical suffixes")
    
    na_counts = worklist_df[worklist_df['Source Well'].isna()]['Liquid Name'].value_counts()
    if not na_counts.empty:
        logger.warning(f"Failed transfers found: {na_counts.to_dict()}")

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
            [worklist_name,                     # A1
            '1.10.1.178',                       # B1
            user,                               # C1
            current_time.strftime("%d.%m.%Y"),  # D1
            current_time.strftime("%H:%M"),     # E1
            '',                                 # F1
            '',                                 # G1
            ''],                                # H1
            [source_plate_type,                 # A2
            source_name,                        # B2
            '',                                 # C2
            '',                                 # D2
            target_type,                        # E2
            target_name,                        # F2
            '',                                 # G2
            'Waste Tube'],                      # H2
            ['DispenseToWaste=' + str(dispense_to_waste),   # A3
            'DispenseToWasteCycles=3',                      # B3
            'DispenseToWasteVolume=5e-8',                   # C3
            'UseDeionisation=' + str(deionisation),         # D3
            'OptimizationLevel=' + optimization,            # E3
            'WasteErrorHandlingLevel=Ask',                  # F3
            'SaveLiquids=Ask',                              # G3
            '']                                             # H3
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

def generate_worklist(
    input_file: Path, 
    output_folder: Path, 
    plate_size: int = '1536', 
    parallelisation: ParallelisationType = ParallelisationType.IMPLICIT,
    header_config: dict = None
) -> str:
    """
    Generate iDot worklist files from input data.
    
    Args:
        input_file (Path): Path to input Excel file containing plate data
        output_folder (Path): Directory where output files will be saved
        plate_size (int): Microplate format (96, 384 or 1536 wells)
        parallelisation (ParallelisationType): Type of parallelisation strategy
            - "implicit": Uses parallel + sequential transfers (default)
            - "explicit": Uses sequential transfers + clean labels
            - "none": Uses only sequential transfers
        header_config (dict): Dictionary containing header configuration parameters
        
    Returns:
        str: Success message with generated output filenames
        
    Raises:
        ValueError: If plate_size is not 96, 384 or 1536 or invalid parallelisation type
    """
    # Setup logger
    setup_logging(output_folder, Path(input_file.name).stem)
    logger.info(f"Starting worklist generation for {input_file}")
    logger.info(f"Configuration - Plate size: {plate_size}, Parallelisation: {parallelisation}")

    # Input validation
    if not input_file or not output_folder:
        return "Please select input file and output folder", None
    if not Path(input_file.name).exists():
        return f"Input file not found: {input_file.name}", None
    if not Path(output_folder).exists():
        return f"Output folder not found: {output_folder}", None
    if not isinstance(parallelisation, ParallelisationType):
        raise ValueError(f"Invalid parallelisation type. Must be one of {[p.value for p in ParallelisationType]}")

    # Read and process dataframes
    dataframes = read_excel_sheets(input_file.name)
    logger.debug(f"Loaded dataframes: {list(dataframes.keys())}")

    # Configure row dictionary based on plate size
    if plate_size == 1536:
        ROWDICT = DEFAULT_ROWDICT
    elif plate_size in [384, 96]:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ROWDICT = dict(zip(letters, letters))
    else:
        raise ValueError("Invalid plate size. Please select 96, 384, or 1536.")

    # Melt dataframes and simplify
    melted_dataframes = {
        sheet_name: melt_and_combine(df, ROWDICT) 
        for sheet_name, df in dataframes.items()
    }
    data_dict = simplify_input_data(melted_dataframes, plate_size)
    logger.debug(f"Data shapes - Source: {data_dict['source'].shape}, Target: {data_dict['target'].shape}")

    # Generate worklist based on parallelisation strategy
    if parallelisation == ParallelisationType.IMPLICIT:
        idot_wl, na_count = create_iDot_worklist(
            data_dict, 
            use_parallel=True,
            clean_labels=False
        )
    elif parallelisation == ParallelisationType.EXPLICIT:
        idot_wl, na_count = create_iDot_worklist(
            data_dict, 
            use_parallel=False,
            clean_labels=True
        )
    else:  # ParallelisationType.NONE
        idot_wl, na_count = create_iDot_worklist(
            data_dict, 
            use_parallel=False,
            clean_labels=False
        )
    
    # Save outputs
    base_name = os.path.splitext(os.path.basename(input_file.name))[0]
    excel_output = os.path.join(output_folder, f"idot_worklist_{base_name}.xlsx")
    csv_output = os.path.join(output_folder, f"idot_worklist_{base_name}.csv")
    
    # Add empty columns for consistency with iDot format
    main_columns = idot_wl.columns.tolist()
    empty_cols = [''] * 3
    csv_headers = main_columns + empty_cols
    for i in range(3):
        idot_wl[f'__empty_{i}'] = ''
    
    # Save files
    idot_wl.to_excel(excel_output, index=False, header=csv_headers)
    idot_wl.to_csv(csv_output, index=False, header=csv_headers)
    logger.info(f"Generated output files: {excel_output}, {csv_output}")

    # Add headers
    add_headers(excel_output, csv_output, **header_config)
    
    return (f"Worklist generated: {os.path.basename(excel_output)} and {os.path.basename(csv_output)}", idot_wl)