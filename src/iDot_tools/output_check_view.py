import numpy as np
import pandas as pd
import random
from .ui_utils import generate_distinct_colors, style_cell
from .constants import ROWDICT

def style_plate_old(plate_df):
    unique_values = plate_df.values.flatten()
    unique_values = [x for x in unique_values if pd.notna(x)]
    colors = generate_distinct_colors(len(unique_values))
    color_map = dict(zip(unique_values, colors))
    
    return plate_df.style.applymap(lambda x: style_cell(x, color_map, style_type='id'))
def empty_plate(size, value_col_name):
    plate_dims = {
        '96': ('H', 12),
        '384': ('P', 24),
        '1536': ('ZF', 48)
    }
    
    last_row, num_cols = plate_dims[size]
    
    if size == '1536':
        row_labels = list(ROWDICT.keys())
    else:
        row_labels = [chr(i) for i in range(ord('A'), ord(last_row) + 1)]
        
    col_labels = list(range(1, num_cols + 1))
    
    empty_df = pd.DataFrame(
        [(row, col, '') for row in row_labels for col in col_labels],
        columns=['idot_row', 'idot_col', value_col_name]
    )
    return empty_df


def plate_view(worklist, value_col='Source Well', plate_size='96'):
    # Extract row and column info
    rev_rowdict = {v: k for k, v in ROWDICT.items()}
    df = worklist.copy()
    
    # Extract row and column info
    if plate_size == '1536':
        df['idot_row'] = df['Target Well'].astype(str).str.extract('([A-Za-z]+)').iloc[:,0].map(rev_rowdict)
    else:
        df['idot_row'] = df['Target Well'].astype(str).str.extract('([A-Za-z]+)').iloc[:,0]
    df['idot_col'] = df['Target Well'].astype(str).str.extract('(\d+)').astype(int)

    # Create empty plate with all combinations
    complete_data = empty_plate(plate_size, plate_size)
    
    # Remove existing combinations from complete_data before merge
    merge_key = df[['idot_row', 'idot_col']].apply(tuple, axis=1)
    complete_key = complete_data[['idot_row', 'idot_col']].apply(tuple, axis=1)
    complete_data = complete_data[~complete_key.isin(merge_key)]
    # Concatenate instead of merge to avoid duplicate columns
    filled_data = pd.concat([complete_data, df[['idot_row', 'idot_col', value_col]]])
    
    # Pivot to create plate view
    plate_matrix = filled_data.pivot(
        index='idot_row',
        columns='idot_col',
        values=value_col
    )
    plate_matrix = plate_matrix.loc[plate_matrix.index != 'idot_col']
    return plate_matrix



def visualise_plate_output(worklist_df: pd.DataFrame, plate_size: int) -> tuple:
    """
    Creates styled plate visualization from worklist DataFrame
    
    Args:
        worklist_df: pandas DataFrame containing worklist data
        plate_size: Size of the plate (96, 384, or 1536)
        
    Returns:
        tuple: (plate_view_html)
    """
    if worklist_df is None:
        return None, None

    plate_matrix = plate_view(worklist_df, plate_size=str(plate_size))
    print(plate_matrix)
    styled_plate = style_plate(plate_matrix)
    
    return (styled_plate.to_html(), )
