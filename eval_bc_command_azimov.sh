python -m scripts.eval_bc --domain=prl_ur5 --top_dir=/scratch/azimov/malakuja/data/rrlfd/ --dataset_origin=real_robot --demo_task=Push --input_type=rgb --nobinary_grip_action --noearly_closing --nogrip_action_from_state --action_norm=zeromean_unitvar --signals_norm=zeromean_unitvar --eval_seed=5000 --increment_eval_seed --visible_state=gripper_pos --max_demos_to_load=50 --noval_full_episodes --dataset=pushing_demos_v2_rgb_e56_shuffled_g128_2022-08-10T10-44-20_12-05-23 --dataset_in_ram --image_size=128 --test_set_size=5 --exp_id=e4_vels_nostops --job_id=without_vels_1 --test_set_start=50 --seed=0 --grayscale=True --load_saved --sim --offline_dataset_path=/scratch/azimov/malakuja/data/rrlfd/pushing_demos_v0_2022-08-10T10-44-20.pkl  # try using dataset in place of env
