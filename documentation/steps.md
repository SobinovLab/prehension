# Processing prehension data

Three sections: data processing, validation, and scaling of musculoskeletal model.

## Data processing order:

### 1 Preprocessing

1. `create_meta.py`
        -- creates session, object and DOF meta files, that are needed for the later steps. Finds the good trials and gets synchronization time periods. Some sessions will have duplicate trials in the beginning, which need to be fixed by creating a copy of session log and REMOVING the starting trials from it that ran twice. These errors only appeared during preliminary data recording on Mojito, and currently those trials are not salvageable. Those trials/sessions can be identified by having multiple trial_logs in sensors directory.

Steps 2 (pressure sensors) and 3 (kinematics) can be completed in either order, but both of them need to be completed before Step 4 (kinematics and pressure sensors together).

### 2 Pressure sensors

1. `preprocessPressureSensorData.m`
        -- transforms pressure sensor data from Tekscan proprietary format into calibrated TSMs. Needs to have session folders specified. Has multiple versions - for each animal.

    * `rename_folders.py`
        -- if the data was exported in the old style (csv's to 'processed_sensors'), this script can rename the directory to 'transformed_sensors'.

2. `filter_pressure_sensors.py`
        -- filters the pressure sensors, attempting to remove electrical noise.

### 3 Kinematics

1*. `compress_session_cameras.py`
        -- makes videos out of images.
        -- Deprecated starting 2023-03 - cameras are recording videos. Needs to be run for failed trials from old sessions still.

2. `analyze_videos.py`
        -- runs machine vision/deeplabcut on the images. Prior to this step, a network needs to be trained. You can use `--dont_analyze --make_videos` flags to inspect the quality of the labeled points.

3. `calibration.py`
        -- all data after the preliminary has extrinsic calibration of cameras done on a per-session basis. Run this function to run extrinsic calibration NCams scripts for all sessions (use `--run_extrinsic_calibration`). Use `--relocate` option to copy the calibration files from `cameras/calibration` into the correct location (`calibration/extrinsic`). 

4. `triangulate.py`
        -- triangulates the points from 2D to 3D using calibrations and creates inverse kinematics command files for OpenSim. If the monkey's left arm was used in the experiment, it will also reflect the points along an axis and switch left and right sensor naming. All following steps will be treating the data as a right-handed animal.

For the following steps, a scaled OpenSim model needs to exist for the animal. See the Scaling section for details.

5. Estimate the static position of animal's thorax. You have two options based on the visibility of proximal (thorax) markers. If they are visible, use step 1, if not, use 2.

    1. Using the markers.
    
        1. `inverse_kinematics.py --base`
                -- runs OpenSim on IK files and produces time-varying position of the thorax.
    
        2. `find_static_thorax_position.py`
                -- create model with fixed thorax position.
    
    2. Using manually-labelled thorax position for thorax for each extrinsic calibration. 
    
        1. `mark_base.py` 
                -- Automatically identifies all different (extrinsic) calibrations associated with all session, and for each allows to select a session and trial to label of landmarks on the body. Then automatically triangulates them, runs inverse kinematics, and creates a session-specific model.

6. `inverse_kinematics.py`
        -- runs OpenSim on IK files and produces generalized coordinates for the model (joint angles).

### 4 Kinematics and pressure sensors together

1. `process_and_align_data.py`
        -- creates processed and aligned to grasp data synchronized between joint angles and pressure sensors.

    1. `inspect_object_endpoints.py`
            -- shows variability in the locations of thorax (base body) and pressure sensors during grasp.

    2. `--make_plots --processes 1`
            -- option will show some synchronization and DOF/PS information.

2. `prepare_mujoco_model.py`
        -- creates a session-specific MuJoCo model with only activated pressure sensels present from a generic MuJoCo model.

3. `automatically_match.py`
        -- finds matches between hand segments and activated pressure sensor sensels for each frame. Applies adjustments if they are specified in `adjustment_files.csv`.

4. `export_digit_forces.py`
        -- exports the matched contact information into a easily readable format.

Optional steps:

1. Adjust the final position of the pressure sensors. If the tracking of the object is suboptimal, it can be manually adjusted for all trials. Since many issues with 3D tracking have been resolved, these adjustments are not needed. If desired, the following steps should be run after 2 (`prepare_mujoco_model.py`) and before 3 (`automatically_match.py`).

    1. `find_optimal_frames.py`
            -- find optimal frames to use for adjusting the relative position of sensors to the thorax. Since adjustment is not needed, step can be skipped.
    
    2. `make_adjustment.py <TRIAL>`
            -- opens an optimal frame from the specified trial and allows adjustment of the final position of the pressure sensor. Since adjustment is not needed, step can be skipped.


### 5 Kinetics

Coming soon...


## Validation

This section describes the steps needed to take for validation. Currently it focuses on the forces, but in the future will have parts on joint angles, too.

1. `compare_masked_forces.py --find_good --find_good_n <NUMBER>`
        -- prints out lists of trials that were successful and had only a single grasp marked in manual log as good candidates for manual labeling and validation.

2. `manually_label_forces.py <SESSION> <TRIAL>`
        -- a gui that allows manually matching forces with digits.

3. `compare_masked_forces.py`
        -- compares the automatic vs manual matching to validate the data processing.

    1. `compare_digit_forces.py`
            -- older script, that compares directly the measured values of forces.

## Scaling process order:

The purpose of scaling is to create an OpenSim model with proportions matching those of a monkey, and virtual marker locations matching monkey's tattoos or other landmarks tracked by machine vision algorithms. Triangulation step needs to be completed before scaling can be done. The scaling requires human input and judgement, and needs to be done once per monkey arm.

1. Copy the [default OpenSim model](osim_models/default_model/RightArmAndHand_NoMuscles.osim) and the associated Geometry from this repository into the opensim_models directory next to the sessions folders. Make a copy of it with `_Scaled` suffix (`RightArmAndHand_NoMuscles.osim` to `RightArmAndHand_NoMuscles_Scaled.osim`). This will be the model that will be scaled to the size of the model, and from here on when referring to the model, I will be talking about the \_Scaled file. If you have a model for the previous macaque, you can use that, since it will be closer in size to the target, although it is easier to move markers on a larger model.

2. Move markers on the model to match the specific locations of the markers/tattoos on the animal. Use videos, pictures from tattoo sessions, and marked videos for guidance (TODO make general automatic function to label sessions). Check that the markers that were created by DLC are present in the model.

3. Choose a good session. Use experimenters' log and quality of video recordings as guidance.

4. Choose a good trial within that session. Preferably where the monkey grasps once and all digits are clearly visible.

5. Run `create_scaling_files.py <SESSION> <TRIAL>`. Do not close the command window, it will be needed in the future to transfer the inverse kinematic position of the arm into the default posture of the scaled model. This generates initial .trc, inverse kinematic and scaling files for the specified trial.

6. (optional) If the model is still human-sized, scale it down using manual scaling.

    1. Load the OpenSim model into the OpenSim.

    2. Open Scaling tool. Select 'Scale Model' and 'Preserve mass distribution during scale'. Deselect 'Adjust Model Markers'.

    3. Switch to the 'Scale Factors' tab. Highlight all bodies except for pressure sensor ones ('PS', 'LPS', 'RPS'). Select 'Use manual scales' and scale all segments to 0.6.
    
    4. Click 'Run'. The newly appeared model might look broken, it does not matter. Right-click on it in the left-hand side menu and 'Save As' over the previous model. Do not 'Save', as it will create a new model.
    
    5. Close OpenSim.

7. Sometimes (in most cases after preliminary Mojito's data) the proximal markers will be obstructed and not visible for inverse kinematics and scaling. These data will benefit from semi-manually estimating the position of the thorax instead of relying on `base` inverse kinematics. 

    1. To do that, run `py mark_base.py --session <SESSION>`. It will identify the relevant camera calibration, and ask the user to label several points on clavicles and thorax, after which it will triangulate them and run inverse kinematics producing static thorax position. 
    
    2. It is essential that the OpenSim's model thorax is unlocked relative to the ground, otherwise the script will not be able to adjust it. 
    
    3. For this animal, in the Kinematics section of Automatic process order, you will have to run this script for each extrinsic calibration (1 per session).
    
    4. Transfer the posture of the locked thorax from the session-specific file (e.g. `RightArmAndHand_NoMuscles_Scaled_locked_2022_03_01_Set1.osim`) into general scaled file (`RightArmAndHand_NoMuscles_Scaled.osim`). Alternatively, just rename the file.

8. Inverse kinematics.

    1. Load the OpenSim model into OpenSim.

    2. Open the Inverse Kinematic tool (Tools->Inverse Kinematics...) and load the `trial<TRIAL>_IK.xml` config from the 'scaling' folder.
    
    3. Click 'Run'. 
    
        * If you get the 'Optimizer failed' error, open the XML file and lower the accuracy by an order of magnitude (e.g. from 1e-5 to 1e-4), close OpenSim and restart the Inverse kinematics section.
        
        * If you see the model assume a weird pose, inspect the marker positions using 'Preview Experimental Data' option in File menu on the .trc file. If you see some markers far from the others, or in places where they do not make sense, disable them in the XML file (switch 'apply' tag from true to false). Close OpenSim and restart the Inverse kinematics section.
        
    4. Close OpenSim.

9. `create_scaling_files.py <SESSION> <TRIAL> --transfer_position` -- makes the average posture from the produced inverse kinematics file the default posture of the OpenSim model.

10. Scaling.

    1. Load the OpenSim model into OpenSim.
    
    2. Open the Scaling tool (Tools->Scale Model...) and load the `trial<TRIAL>_SC.xml` config from the 'scaling' folder.
    
    3. Switch to the Scale Factors tab. If this is the first time running the Scaling tool, all Measurements will be absent. Use the markers available in the kinematics during this period, the output of `create_scaling_files` - which markers are available attached to each segment, and common sense. After the measurements have been assigned, save the scaling file, making a copy with `_updated` in the name so you won't occasionally overwrite it by running `create_scaling_files`, and so that next iteration is easier.
    
    * Since the macaque proportions are slightly different, you might want to scale some bones (e.g., ulna, radius, humerus) differently on different axes. It is mostly cosmetic.
    
    * Some markers on some segments might be lacking, so either use other segments as reference, or external measurements of the segment.
    
    * There is no easy way to extract distance between markers on a model in a certain posture, so a way to use the measurements done on the animal directly is to change the markers that a measurement is attached to. Upon change, OpenSim GUI will print out the distance between those markers, from which the manual scaling factor can be calculated. Do not hesitate to add or move markers to utilize all available direct segment measurements.

11. If you are not satisfied with result, goto 8. If all scaling factors are 97-103% and the model looks reasonable (in accordance with videos and common sense), continue.

12. If the model has a session-specific name, rename it to the generic one (`opensim_models/RightArmAndHand_NoMuscles_Scaled.osim`).

13. `transform_osim_model.py <SESSION>` - transform OpenSim model into a MuJoCo model. We can use one monkey-specific MuJoCo model, as the ground-thorax DOFs are still kinematically controlled during the matching.
