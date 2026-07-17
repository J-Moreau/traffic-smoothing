# 4DVar for Traffic Smoothing

Code for our [ITSC2026 paper](https://hal.science/hal-05649935/document) on traffic data assimilation with probe vehicles.

![Traffic Flow Visualization](output_plots/fourdvar_mobile-century_20pc.gif)

*Traffic state reconstruction with 4DVar*

## Setup

Install [uv](https://github.com/astral-sh/uv) then run

```
uv sync
```

Download data from

* [NGSIM](https://data.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/)
```sh
mv Next_Generation_Simulation__NGSIM__\
Vehicle_Trajectories_and_Supporting_Data_20250312.csv data/ngsim/
```

* [Mobile Century](https://traffic.berkeley.edu/project/downloads/mobilecenturydata)
```sh
cp -r MobileCentury_data_final_ver3/NB_Veh_files data/mobilecentury/
cp -r MobileCentury_data_final_ver3/SB_Veh_files data/mobilecentury/
```

## Demonstration notebook

[notebooks/demo.ipynb](notebooks/demo.ipynb)

## Reproduce results

We use Weights and Biases for logging
```sh
uv run python -m traffic_models.experiments.sweep
```

Inspect results with [this notebook](notebooks/sweep_results.ipynb)
