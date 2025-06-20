#!/usr/bin/env python3

import os
import sys
from .utils import read_excel_sheets, melt_and_combine, create_iDot_worklist
from .constants import ROWDICT


def main(source_folder, source_file):
    file_path = os.path.join(source_folder, source_file)
    dataframes = read_excel_sheets(file_path)
    for sheet_name, df in dataframes.items():
        print(f"File: {sheet_name}, Sheet: {sheet_name}")

    melted_dataframes = {}

    for sheet_name, df in dataframes.items():
        melted_dataframes[sheet_name] = melt_and_combine(df, ROWDICT)

    idot_wl, na_count = create_iDot_worklist(melted_dataframes)

    # Write file
    base_name = os.path.splitext(source_file)[0]
    excel_output = f"idot_worklist_{base_name}.xlsx"
    csv_output = f"idot_worklist_{base_name}.csv"
    idot_wl.to_excel(os.path.join(source_folder, excel_output), index=False)
    idot_wl.to_csv(os.path.join(source_folder, csv_output), index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <source_folder> <source_file>")
        sys.exit(1)

    source_folder = sys.argv[1]
    source_file = sys.argv[2]

    main(source_folder, source_file)
