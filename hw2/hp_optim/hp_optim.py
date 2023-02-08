import subprocess
import json
import optuna
import math
import itertools

CMD_1 = "\
python run_hw2.py --env_name CartPole-v0 -n 100 --data_path data_tmp \
-b {batch_size} {dsa} {rtg} --exp_name q1_sb_no_rtg_dsa \
"

CMD_2 = "\
python run_hw2.py --env_name InvertedPendulum-v4 --data_path data_tmp \
    --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b {batch_size} -lr {lr} -rtg \
    --exp_name q2_b{batch_size}_r{lr}\
"

CMD_4 = "\
python run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --data_path data_q4 \
    --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {lr} -rtg --nn_baseline \
    --exp_name q4_search_b{batch_size}_lr{lr}_rtg_nnbaseline \
"

CMD_5 = "\
    python run_hw2.py  \
    --env_name Hopper-v4 --ep_len 1000 \
    --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \
    --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda {lambda} \
    --exp_name q5_b2000_r0.001_lambda{lambda}\
"

# CMD = "\
# cat test_{hp}.o \
# "

HP_VALS = range(13)
data = {}

# HP_VALS = [11, 12]
# with open('hp_curve_hopper.json') as f:
#     data = json.load(f)

def get_run_results(cmd, hp_vals):
    print(f"\n===== RUN with {hp_vals=} =====")
    run_result = subprocess.run(cmd.format(**hp_vals), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    run_out = run_result.stdout.decode('utf-8')
    data = {}
    for line in run_out.split('\n'):
        parts = line.split(' : ')
        if len(parts) == 2:
            data[parts[0]] = float(parts[1])
    for key, val in data.items():
        print(key, ':', val)
    return data

def optim_objective_1(trial):
    hp_vals = {
        'dsa': trial.suggest_categorical('dsa', ['', '-dsa']),
        'rtg': trial.suggest_categorical('rtg', ['', '-rtg']),
        'batch_size': trial.suggest_int('batch_size', 100, 10000, log=True),
    }
    return get_run_results(CMD_1, hp_vals)['Eval_AverageReturn']

def optim_objective_2(trial):
    hp_vals = {
        'lr': trial.suggest_float('lr', 1e-4, 0.1, log=True),
        'batch_size': trial.suggest_int('batch_size', 1000, 25000, log=True),
    }
    avg_r = get_run_results(CMD_2, hp_vals)['Eval_AverageReturn']
    print("Real return:", avg_r)

    BATCH_COEFF = 1 # 1 for equilibrium, >1 for smaller batch_size, <1 for larger lr
    avg_r += math.log(hp_vals['lr'])
    avg_r -= math.log(hp_vals['batch_size']) * BATCH_COEFF
    return avg_r

def optim_objective_5(trial):
    hp_vals = {'lambda': trial.suggest_float('lambda', 0, 1)}
    return get_run_results(CMD_5, hp_vals)['Eval_AverageReturn']

def optim_objective_4(trial):
    hp_vals = {
        'lr': trial.suggest_float('lr', 0.001, 0.05, log=True),
        'batch_size': trial.suggest_int('batch_size', 5000, 50000, log=True),
    }
    return get_run_results(CMD_4, hp_vals)['Eval_AverageReturn']


def main_optuna():
    # objective = optim_objective_1
    # objective = optim_objective_2
    objective = optim_objective_4
    # objective = optim_objective_5

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print()
    print("BEST PARAMS")
    print(study.best_params)

def main_gridsearch():
    cmd = CMD_4
    hp_values = {
        'lr': [0.005, 0.01, 0.02],
        'batch_size': [10000, 30000, 50000],
    }
    # cmd = CMD_5
    # hp_values = {
    #     'lambda': [0,0.95,0.99,1],
    # }

    keys = list(hp_values.keys())
    best_return, best_hps, i_best = None, None, None
    for i_cur, hp_take_val in enumerate(itertools.product(*[hp_values[k] for k in keys])):
        hps = {k: v for k, v in zip(keys, hp_take_val)}
        print(f"************ Experiment {i_cur} with parameters {hps} ************")
        data = get_run_results(cmd, hps)
        val_return = data['Eval_AverageReturn']
        if best_return is None or val_return > best_return:
            best_return, best_hps, i_best = val_return, hps, i_cur

        print()
        print(f"Return value for iteration {i_cur} is {val_return}")
        print(f"Current best is iteration {i_best} with return {best_return} and configuration {best_hps}")
        print()


if __name__ == "__main__":
    # main_optuna()
    main_gridsearch()
