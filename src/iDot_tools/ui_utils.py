import pandas as pd
import numpy as np
import random
from typing import Dict, Any

def generate_distinct_colors(n: int) -> list:
    """Generate visually distinct colors using HSV color space."""
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
        colors.append(f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}')
    return colors

def is_dark_color(hex_color):
    """Determine if a color is dark based on its RGB values."""
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Calculate luminance (perceived brightness)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return luminance < 0.5

def style_ids(val, item2col):
    """Style function for ID dataframes."""
    if pd.notna(val) and val in item2col:
        bg_color = item2col[val]
        text_color = '#ffffff' if is_dark_color(bg_color) else '#000000'
        return f'background-color: {bg_color}; color: {text_color}'
    return ''

def style_vol(val, id_df, item2col):
    """Style function for volume dataframes based on corresponding ID dataframe."""
    if pd.notna(val) and pd.notna(id_df) and id_df in item2col:
        bg_color = item2col[id_df]
        text_color = '#ffffff' if is_dark_color(bg_color) else '#000000'
        return f'background-color: {bg_color}; opacity: 0.95; color: {text_color}'
    return ''

def style_dataframe(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Style dataframes with color coding based on IDs and volumes."""
    source_df = data_dict['source_id']
    source_items = pd.unique(source_df.iloc[:, 1:].values.ravel())
    items = [x for x in source_items if pd.notna(x)]
    
    colors = generate_distinct_colors(len(items))
    item2col = dict(zip(items, colors))
    
    styled_source_id = data_dict['source_id'].style.apply(
        lambda x: pd.Series([''] + [style_ids(v, item2col) for v in x[1:]], index=x.index),
        axis=1
    ).hide(axis='index')
    
    styled_target_id = data_dict['target_id'].style.apply(
        lambda x: pd.Series([''] + [style_ids(v, item2col) for v in x[1:]], index=x.index),
        axis=1
    ).hide(axis='index')
    
    styled_source_vol = data_dict['source_vol'].style.apply(
        lambda x: pd.Series(
            [''] + [style_vol(v, data_dict['source_id'].iloc[x.name, i+1], item2col) 
                   for i, v in enumerate(x[1:])],
            index=x.index
        ),
        axis=1
    ).hide(axis='index')
    
    styled_target_vol = data_dict['target_vol'].style.apply(
        lambda x: pd.Series(
            [''] + [style_vol(v, data_dict['target_id'].iloc[x.name, i+1], item2col)
                   for i, v in enumerate(x[1:])],
            index=x.index
        ),
        axis=1
    ).hide(axis='index')
    
    legend_df = pd.DataFrame([item2col.keys()], columns=item2col.keys())
    legend_df = legend_df.style.apply(
        lambda x: pd.Series([style_ids(v, item2col) for v in x], index=x.index),
        axis=1
    ).hide(axis='index').hide(axis='columns')

    return {
        'source_id': styled_source_id,
        'source_vol': styled_source_vol,
        'target_id': styled_target_id,
        'target_vol': styled_target_vol,
        'legend': legend_df,
        'color_mapping': item2col
    }