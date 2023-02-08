
# Experiment 1 (CartPole)

# python run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
#     -dsa --exp_name q1_sb_no_rtg_dsa
# python run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
#     -rtg -dsa --exp_name q1_sb_rtg_dsa
# python run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
#     -rtg --exp_name q1_sb_rtg_na
# python run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
#     -dsa --exp_name q1_lb_no_rtg_dsa
# python run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
#     -rtg -dsa --exp_name q1_lb_rtg_dsa
# python run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
#     -rtg --exp_name q1_lb_rtg_na

## Experiment 1.2
# python run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
#     --exp_name q1_sb_no_rtg_na
# -> 200

# Experiment 2 (InvertedPendulum)

# # C=0.5
# python run_hw2.py --env_name InvertedPendulum-v4 \
#     --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 2759 -lr 0.09955244631066584 -rtg \
#     --exp_name q2_l0.5
# # -> 1000

# # C=1
# python run_hw2.py --env_name InvertedPendulum-v4 \
#     --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 1097 -lr 0.03283988437943506 -rtg \
#     --exp_name q2_l1
# # -> 1000

# # C=2.5
# python run_hw2.py --env_name InvertedPendulum-v4 \
#     --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 790 -lr 0.029399377942098733 -rtg \
#     --exp_name q2_l2.5
# # -> 1000

# Experiment 3

# python run_hw2.py \
#     --env_name LunarLanderContinuous-v4 --ep_len 1000 \
#     --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 \
#     --reward_to_go --nn_baseline --exp_name q3_b40000_r0.005

# Experiment 4

python run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.005 -rtg --nn_baseline \
    --exp_name q4_search_b10000_lr0.005_rtg_nnbaseline

# Experiment 5

# python run_hw2.py \
#     --env_name Hopper-v2 --ep_len 1000 \
#     --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \
#     --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda 0 \
#     --exp_name q5_b2000_r0.001_lambda0