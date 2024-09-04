# Prehension data analysis

[description]

## Prerequisites

`tsm` from https://github.com/nishbo/timed_sparse_matrix

`reporting_pool` from https://github.com/nishbo/reporting_pool

`deeplabcut` - only for video analyses (`analyze_videos`)

`ffmpegio` - only for video compression (`compress_session_cameras`)

`opensim` for opensim usage (`inverse_kinematics`, `execute_opensim_ik.py`)

`ncams` from https://github.com/CMGreenspon/NCams (only for `calibration`, `create_scaling_files`, `mark_base`, 
    `triangulate`, `predict_points_jarvis`)

`jarvis` for new video analyses (`predict_points_jarvis`, `train_jarvis`)

`torch` for (`predict_points_jarvis`, `train_jarvis`)

`pythonnet` for `preprocess_pressure_sensors`

`O2MConverter` for `transform_osim_model`


## Installation

`py -3.11 -m pip install -e .`


## License

## Authors
