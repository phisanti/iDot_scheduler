import numpy as np
import pandas as pd
import random
from .ui_utils import generate_distinct_colors, style_cell
from .utils import empty_plate
from .constants import ROWDICT


def style_plate(plate_df: pd.DataFrame):
    """Style plate DataFrame with distinct colors for each unique value.
    Args: plate_df (DataFrame) - Input plate data
    Returns: Styled DataFrame with color formatting"""
    unique_values = plate_df.values.flatten()
    unique_values = [x for x in unique_values if pd.notna(x)]
    colors = generate_distinct_colors(len(unique_values))
    color_map = dict(zip(unique_values, colors))

    return plate_df.style.applymap(lambda x: style_cell(x, color_map, style_type="id"))


def plate_view(
    worklist: pd.DataFrame, value_col: str = "Source Well", plate_size: str = "96"
) -> pd.DataFrame:
    """Transform worklist data into plate matrix format.
    Args: worklist (DataFrame) - Input worklist, value_col (str) - Column to display, plate_size (str) - Plate dimensions
    Returns: DataFrame in plate matrix layout"""
    # Extract row and column info
    rev_rowdict = {v: k for k, v in ROWDICT.items()}
    df = worklist.copy()

    # Extract row and column info
    if plate_size == "1536":
        df["idot_row"] = (
            df["Target Well"]
            .astype(str)
            .str.extract("([A-Za-z]+)")
            .iloc[:, 0]
            .map(rev_rowdict)
        )
    else:
        df["idot_row"] = (
            df["Target Well"].astype(str).str.extract("([A-Za-z]+)").iloc[:, 0]
        )
    df["idot_col"] = df["Target Well"].astype(str).str.extract("(\d+)").astype(int)

    # Create empty plate with all combinations
    complete_data = empty_plate(plate_size, "")

    # Remove existing combinations from complete_data before merge
    merge_key = df[["idot_row", "idot_col"]].apply(tuple, axis=1)
    complete_key = complete_data[["idot_row", "idot_col"]].apply(tuple, axis=1)
    complete_data = complete_data[~complete_key.isin(merge_key)]
    # Concatenate instead of merge to avoid duplicate columns
    filled_data = pd.concat([complete_data, df[["idot_row", "idot_col", value_col]]])

    # Pivot to create plate view
    plate_matrix = filled_data.pivot(
        index="idot_row", columns="idot_col", values=value_col
    )
    plate_matrix = plate_matrix.loc[plate_matrix.index != "idot_col"]
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
    styled_plate = style_plate(plate_matrix)

    return (styled_plate.to_html(),)
