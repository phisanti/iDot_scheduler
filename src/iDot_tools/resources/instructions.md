# iDot Scheduler Instructions

## Input Requirements

- File format must be **Excel (.xls/.xlsx)**
- Must contain **4 sheets** named exactly:
  1. **source_id**
  2. **source_vol**
  3. **target_id**
  4. **target_vol**

## Data Format
- First row must be just a label with the sheet type (source_id, source_vol, target_id, or target_vol)
- Second row must be **header row** with numbers 1-n (depending on the plate type).
- First column must be labeled **Row**
- Values must be **numeric** in volume sheets
- All IDs must match between source and target sheets. 

## Parallelisation Options

- **Implicit**: Uses both parallel and sequential transfers (default)
- **Explicit**: Uses sequential transfers with cleaned labels (removes "-N" suffixes, e.g. "sample-1" → "sample") 
- **None**: Uses only sequential transfers
- 
## Common Errors

- Missing or misnamed sheets
- Non-numeric values in volume sheets
- Insufficient source volume
- Missing source-target ID matches
- Dead volume violations (< 1µL)

## Workflow

1. Upload Excel file
2. Select output folder
3. Choose plate size
4. Click "Read Files" to validate
5. Click "Generate Worklist" to create schedule
