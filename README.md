#  

# iDot Scheduler Tool

A Python tool for automating pipetting worksheets generation for the iDot liquid dispenser. The tool helps optimize plate layouts and create standardized worklists.

## Installation

pipinstallgit+https://github.com/phisanti/e019_iDot_scheduler

## Usage

Currently, it works as CLI tool. It can be used as follows:

```bash
python ./src/iDot_tools/main.py <source_folder> <input_file>
```

Example:

```bash
python ./src/iDot_tools/main.py ./data/sources/ 'input_example.xlsx'
```

### Input Requirements

1. Source folder must contain:
   * `dict_1536platecol_id.json`: Mapping file for well coordinates
   * Other supporting data files
2. Input Excel file must have these sheets:
   * [source_id](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Source plate well identifiers
   * [source_vol](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Available volumes in source plates
   * [target_id](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Target plate well identifiers
   * [target_vol](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Required volumes in target plates

### Output

The tool generates:

* Excel worklist (.xlsx)
* CSV worklist (.csv)

Both files contain the optimized pipetting instructions for the iDot liquid dispenser.
