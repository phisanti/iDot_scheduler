import pandas as pd
import random
from typing import Tuple
from pathlib import Path

# Local Imports
from .utils import read_excel_sheets


def generate_distinct_colors(n: int) -> list:
    """
    Generate visually distinct colors using HSV color space.

    Args:
        n (int): Number of distinct colors to generate

    Returns:
        list: List of hex color codes
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + random.random() * 0.3
        value = 0.8 + random.random() * 0.2
        # Convert HSV to RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c

        if h < 1:
            rgb = (c, x, 0)
        elif h < 2:
            rgb = (x, c, 0)
        elif h < 3:
            rgb = (0, c, x)
        elif h < 4:
            rgb = (0, x, c)
        elif h < 5:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)

        rgb = tuple(int((color + m) * 255) for color in rgb)
        colors.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
    return colors


def is_dark_color(hex_color):
    """
    Determine if a color is dark based on its RGB values.

    Args:
        hex_color (str): Hex color code (with or without # prefix)

    Returns:
        bool: True if color is dark, False if light

    Raises:
        ValueError: If hex_color is invalid format
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Calculate luminance (perceived brightness)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return luminance < 0.5


def style_cell(val, color_map: dict, style_type: str = "id", ref_id: str = None) -> str:
    """
    Style cells based on type and color mapping.

    Args:
        val: Cell value to style
        color_map (dict): Dictionary mapping IDs to colors
        style_type (str): 'id' or 'volume' to determine styling logic
        ref_id (str, optional): Reference ID for volume styling

    Returns:
        str: CSS styling string
    """
    if style_type == "volume":
        check_val = ref_id
    else:
        check_val = val

    if pd.notna(val) and pd.notna(check_val) and check_val in color_map:
        bg_color = color_map[check_val]
        text_color = "#ffffff" if is_dark_color(bg_color) else "#000000"
        opacity = "opacity: 0.95;" if style_type == "volume" else ""
        return f"background-color: {bg_color}; {opacity} color: {text_color}"
    return ""


def create_color_mapping(source_df: pd.DataFrame) -> dict:
    """
    Create color mapping for unique items in source dataframe.

    Args:
        source_df (pd.DataFrame): Source ID dataframe

    Returns:
        dict: Mapping of unique items to color codes
    """
    unique_items = pd.unique(source_df.iloc[:, 1:].values.ravel())
    valid_items = [x for x in unique_items if pd.notna(x)]
    colors = generate_distinct_colors(len(valid_items))
    return dict(zip(valid_items, colors))


def create_legend(color_map: dict) -> pd.DataFrame.style:
    """
    Create a styled legend dataframe.

    Args:
        color_map (dict): Mapping of items to colors

    Returns:
        pd.DataFrame.style: Styled legend dataframe
    """
    legend_df = pd.DataFrame([color_map.keys()], columns=color_map.keys())
    return (
        legend_df.style.apply(
            lambda x: pd.Series(
                [style_cell(v, color_map, "id") for v in x], index=x.index
            ),
            axis=1,
        )
        .hide(axis="index")
        .hide(axis="columns")
    )


def create_styled_table(
    df: pd.DataFrame,
    color_map: dict,
    is_volume: bool = False,
    ref_df: pd.DataFrame = None,
) -> pd.DataFrame.style:
    """
    Create a styled table with consistent formatting.

    Args:
        df (pd.DataFrame): Data to style
        color_map (dict): Color mapping for styling
        is_volume (bool): Whether table contains volumes
        ref_df (pd.DataFrame, optional): Reference dataframe for volume styling

    Returns:
        pd.DataFrame.style: Styled dataframe
    """
    style_type = "volume" if is_volume else "id"

    # Replace NaN and 0 volumes with empty string to make better visualisation
    styled_df = df.copy()
    if is_volume:
        styled_df = styled_df.replace({0: "", 0.0: "", 0.000: ""})
    styled_df = styled_df.fillna("")

    return styled_df.style.apply(
        lambda x: pd.Series(
            [""]
            + [
                style_cell(
                    v,
                    color_map,
                    style_type,
                    ref_df.iloc[x.name, i + 1] if ref_df is not None else None,
                )
                for i, v in enumerate(x[1:])
            ],
            index=x.index,
        ),
        axis=1,
    ).hide(axis="index")


def visualise_input_data(file_path: Path) -> Tuple[str, str, str, str, str]:
    """
    Process Excel file and return HTML representations.

    Args:
        file_path (Path): Path to Excel file

    Returns:
        Tuple[str, str, str, str, str]: HTML strings for source_id, source_vol,
            target_id, target_vol tables and legend

    Raises:
        FileNotFoundError: If file does not exist
        pd.errors.EmptyDataError: If file contains no data
        ValueError: If data format is invalid
    """
    try:
        dataframes = read_excel_sheets(file_path)
        color_map = create_color_mapping(dataframes["source_id"])

        styled_tables = {
            "source_id": create_styled_table(dataframes["source_id"], color_map),
            "target_id": create_styled_table(dataframes["target_id"], color_map),
            "source_vol": create_styled_table(
                dataframes["source_vol"], color_map, True, dataframes["source_id"]
            ),
            "target_vol": create_styled_table(
                dataframes["target_vol"], color_map, True, dataframes["target_id"]
            ),
        }

        legend = create_legend(color_map)

        return (
            styled_tables["source_id"].to_html(),
            styled_tables["source_vol"].to_html(),
            styled_tables["target_id"].to_html(),
            styled_tables["target_vol"].to_html(),
            legend.to_html(),
        )
    except FileNotFoundError:
        raise ValueError("Excel file not found")
    except pd.errors.EmptyDataError:
        raise ValueError("Excel file is empty")
