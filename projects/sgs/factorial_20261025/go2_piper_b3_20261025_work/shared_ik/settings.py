"""One controller configuration for B2/B3. Changes require a new shared version."""
VERSION = 'go2-piper-arm-only-ik-v2.0.0-candidate'
LEG_NAMES = [f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
             for joint in ('hip', 'thigh', 'calf')]
ARM_NAMES = [f'joint{i}' for i in range(1, 7)]
IK = dict(iterations=12, damping=.02, orientation_length_m=.25,
          max_iteration_step_rad=.15, tracking_speed_rad_s=2.,
          position_tolerance_m=.01, orientation_tolerance_rad=.05,
          minimum_vertical_projection=.2, foot_radius_m=.022,
          foot_center_body_m=[-.002, 0., 0.], ground_z_m=0.,
          max_abs_position_m=10.)
