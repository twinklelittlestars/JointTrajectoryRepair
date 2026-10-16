# JointTrajectoryRepair

This repository contains the source code and supporting materials for the paper:

**"A Two-Way Street: On Jointly Repairing Positions and Attributes in Trajectory Data."**

Detailed proofs of all theoretical results are provided in [Supplementary Proofs](./Supplementary_Proofs.pdf).

## File Structure

```
JointTrajectoryRepair/
├── code/
│   ├── main.py            # Main entry point
│   ├── TJER.py            # Exact algorithm
│   ├── TJCR.py            # Quadratic-time approximation
│   ├── TJGR.py            # Linear-time approximation
│   └── requirements.txt   # Python dependencies
├── data/                  # Dataset files
├── Supplementary_Proofs.pdf  # Detailed theoretical proofs
└── README.md
```

## Datasets

The experiments are conducted on the following datasets:

- **PVS:** https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets

- **Onboard:** https://www.kaggle.com/datasets/mrisdal/zurich-urban-micro-aerial-vehicle

- **OpenSky:** https://opensky-network.org/datasets/

## Requirements

We recommend using the following environment:

```
Python 3.9
numpy==1.21.5
pandas==1.4.2
scikit-learn==1.0.2
```

Install dependencies via:

```
pip install -r requirements.txt
```

## How to Run

Navigate to the `code` directory and run:

```
python main.py --dataset [DATASET_NAME] --alg [ALGORITHM_NAME] [--w WINDOW_SIZE]
```

## Parameters

### Required

- `--dataset`  
  Dataset name. Options:

  - `pvs`
  - `onboard`
  - `opensky`
  - `custom`

- `--alg`  
  Algorithm name:
  - `TJER` (Exact algorithm)
  - `TJCR` (Quadratic-time approximation)
  - `TJGR` (Linear-time approximation)

### Optional

- `--w`  
  Window size (default: `7`)

### Additional parameters (for `custom` dataset only)

When using `--dataset custom`, you must provide:

- `--path_err`  
  Path to the dirty dataset

- `--path_clean`  
  Path to the clean dataset

- `--pos_cols`  
  Position columns (comma-separated string)  
  Example: `"lat,lon,alt"`

- `--attr_cols`  
  Attribute columns (comma-separated string)  
  Example: `"speed,heading,rpm"`

- `--attr_thresholds`  
  JSON dictionary specifying thresholds for each attribute  
  Example: `'{"speed":3000,"heading":10000,"rpm":200000}'`

## Special Handling for OpenSky

The OpenSky dataset is processed differently from other datasets:

- Data is grouped by **`icao24` (aircraft identifier)**
- Each trajectory is processed independently
- Results are merged back into the full dataset

This ensures correctness for trajectory-level repair.

## Examples

### Run TJCR on PVS

```
python main.py --dataset pvs --alg TJCR
```

### Run TJCR on Onboard

```
python main.py --dataset onboard --alg TJCR
```

### Run TJGR on OpenSky

```
python main.py --dataset opensky --alg TJGR
```

### Run on custom dataset

```
python main.py \
  --dataset custom \
  --alg TJGR \
  --path_err /path/to/dirty.csv \
  --path_clean /path/to/clean.csv \
  --pos_cols "x,y,z" \
  --attr_cols "a,b,c" \
  --attr_thresholds '{"a":1.0,"b":2.0,"c":3.0}'
```
