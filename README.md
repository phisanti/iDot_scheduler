# iDot Scheduler Tool

A Python tool for automating pipetting worksheets generation for the iDot liquid dispenser. The tool helps optimize plate layouts and create standardized worklists.

## Installation

Create a virtual environment before installation:

```bash
conda create -n idot_env python=3.12
conda activate idot_env
```

Install the package from GitHub:

```bash
pip install git+https://github.com/phisanti/iDot_scheduler
```

Requirements: Python 3.9, pandas, gradio, openpyxl, numpy. Note: Install gradio via pip for optimal compatibility.

## Usage

Currently, it works as CLI tool. It can be used as follows:

```bash
conda activate idot_env
idot-scheduler
```

This should launch the GUI:
![iDot Scheduler Demo](docs/example_usage.gif)

Otherwise, click on the local URL (e. g. Running on local URL:  http://127.0.0.1:7860).

### Input Requirements

Input Excel file must have these sheets:

* [source_id](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Source plate well identifiers
* [source_vol](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Available volumes in source plates
* [target_id](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Target plate well identifiers
* [target_vol](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Required volumes in target plates

### Output

The tool generates:

* Excel worklist (.xlsx)
* CSV worklist (.csv)

Both files contain the optimized pipetting instructions for the iDot liquid dispenser.
