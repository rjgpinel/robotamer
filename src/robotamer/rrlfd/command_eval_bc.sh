#!/bin/bash

set -x

python -m rrlfd.eval_bc --top_dir=/home/minttu/data/rrlfd/ --dataset_origin=real_robot --demo_task=Push --input_type=rgb --nobinary_grip_action --noearly_closing --nogrip_action_from_state --action_norm=zeromean_unitvar --signals_norm=zeromean_unitvar --eval_seed=5000 --increment_eval_seed --visible_state=robot --max_demos_to_load=20 --noval_full_episodes --dataset=pushing_demos_v2_rgb_e31_shuffled_g128_2022-08-09T13-31-51 --dataset_in_ram --image_size=128 --test_set_size=5 --exp_id=e3_gripper_test25-30 --job_id=1 --test_set_start=25 --grayscale=True