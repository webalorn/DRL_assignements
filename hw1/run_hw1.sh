# Commands to run

## BC

python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --n_layers 5 --size 256 -lr 0.001 --num_agent_train_steps_per_iter 20000 \
    --batch_size 10000 --eval_batch_size 10000 \
    --video_log_freq -1

python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
    --env_name HalfCheetah-v4 --exp_name bc_HalfCheetah --n_iter 1 \
    --expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl \
    --n_layers 5 --size 256 -lr 0.001 --num_agent_train_steps_per_iter 20000 \
    --batch_size 10000 --eval_batch_size 10000 \
    --video_log_freq -1

python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Hopper.pkl \
    --env_name Hopper-v4 --exp_name bc_Hopper --n_iter 1 \
    --expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
    --n_layers 5 --size 256 -lr 0.001 --num_agent_train_steps_per_iter 20000 \
    --batch_size 10000 --eval_batch_size 10000 \
    --video_log_freq -1

python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Walker2d.pkl \
    --env_name Walker2d-v4 --exp_name bc_Walker2d --n_iter 1 \
    --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
    --n_layers 5 --size 256 -lr 0.001 --num_agent_train_steps_per_iter 20000 \
    --batch_size 10000 --eval_batch_size 10000 \
    --video_log_freq -1

## DAgger

python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant \
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --n_layers 5 --size 256 -lr 0.001 --num_agent_train_steps_per_iter 2000 \
    --batch_size 10000 --eval_batch_size 10000 --n_iter 10 --do_dagger \
    --video_log_freq -1

python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
    --env_name HalfCheetah-v4 --exp_name dagger_HalfCheetah \
    --expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl \
    --n_layers 5 --size 256 -lr 0.001 --num_agent_train_steps_per_iter 2000 \
    --batch_size 10000 --eval_batch_size 10000 --n_iter 10 --do_dagger \
    --video_log_freq -1

python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Hopper.pkl \
    --env_name Hopper-v4 --exp_name dagger_Hopper \
    --expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
    --n_layers 5 --size 256 -lr 0.001 --num_agent_train_steps_per_iter 2000 \
    --batch_size 10000 --eval_batch_size 10000 --n_iter 10 --do_dagger \
    --video_log_freq -1

python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Walker2d.pkl \
    --env_name Walker2d-v4 --exp_name dagger_Walker2d \
    --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
    --n_layers 5 --size 256 -lr 0.001 --num_agent_train_steps_per_iter 2000 \
    --batch_size 10000 --eval_batch_size 10000 --n_iter 10 --do_dagger \
    --video_log_freq -1
