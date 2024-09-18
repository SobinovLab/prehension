### ```timepoints.csv``` column descriptions

For each session processed, a CSV file called `timepoints.csv` is created with the following column headers:

1. `trial_id`: The ID number of the trial.

2. `shoulder_movement_onset`: The time in seconds since the TTL pulse where shoulder movement is significant enough to be considered an onset. This is taken to be the point where the summed and normalized (to [0, 1]) joint velocities exceed 0.25. The joint angles considered are: ```["ra_sh_rot", "ra_shoulder1_r2_d"]```

3. `elbow_movement_onset`: The time where elbow movement is significant enough to be considered an onset. This is taken to be the point where the summed and normalized (to [0, 1]) joint velocities exceed 0.25. The joint angles considered are: ```["ra_el_e_f"]```

4. `wrist_movement_onset`: The time where wrist movement is significant enough to be considered an onset. This is taken to be the point where the summed and normalized (to [0, 1]) joint velocities exceed 0.25. The joint angles considered are: ```["ra_wr_sup_pro","ra_wr_rd_ud","ra_proximal_distal_r1_d", "ra_proximal_distal_r3_d"]```

5. `fingers_movement_onset`: The time where finger movement is significant enough to be considered an onset. This is taken to be the point where the summed and normalized (to [0, 1]) joint velocities exceed 0.25. The joint angles considered are: ```["ra_cmc1_f_e","ra_cmc1_opp","ra_cmc1_ad_ab","ra_mcp1_e_f","ra_ip1_e_f","ra_mcp2_e_f","ra_mcp2_ad_ab","ra_pip2_e_f","ra_dip2_e_f","ra_mcp3_e_f","ra_mcp3_rd_ud","ra_pip3_e_f","ra_dip3_e_f","ra_mcp4_e_f","ra_mcp4_ad_ab","ra_pip4_e_f", "ra_dip4_e_f","ra_mcp5_e_f","ra_mcp5_ad_ab","ra_pip5_e_f","ra_dip5_e_f"]```

6. `maximum_aperture`: The time of the maximum aperture angle between the thumb and index finger base joint.
The maximum aperture is calculated to be the time when the index finger joint angle (```ra_mcp2_e_f```) and thumb joint angle (```ra_mcp1_e_f```) have the greatest sum.

7. `grasp_start`: The time where the first grasp event is detected. This is defined as the point where the summed, normalized (to [0,1]) force threshold exceeds the threshold ```ONSET_FORCE_THRESH = 0.1``` AND remains above that threshold for a duration greater than ```MIN_GRASP_TIME_S = 0.2``` seconds. Note: the point at which we exceed the force threshold is taken as the `grasp_start` value.

8. `fingers_static`: The time after `grasp_start` where the normalized aggregate finger velocity (defined in 5) drops below ```FING_STATIC_JA_THRESH = 0.4```

9. `release_start`: The time after `fingers_static` where the normalized aggregate finger velocity (defined in 5) exceed ```FING_STATIC_JA_THRESH = 0.4```.

10. `release`: The time after grasp where the normalized total force drops below ```OFFSET_FORCE_THRESH = 0.1```

11. `hand_retreated`: The time, after release, where the difference between the vector of pregrasp positions returns to a local minimum. The joint angles considered are: ```PREGRASP_POSITION_JAS = ["ra_sh_elv_angle","ra_sh_elv","ra_shoulder1_r2_d", "ra_sh_rot"]```

12. `regrasp (boolean)`: True if there is more than one grasp event detected in (7).



