import subprocess
import json
import optuna
import math

CMD_1 = "\
python run_hw2.py --env_name CartPole-v0 -n 100 --data_path data_tmp \
-b {batch_size} {dsa} {rtg} --exp_name q1_sb_no_rtg_dsa \
"

CMD_2 = "\
python run_hw2.py --env_name InvertedPendulum-v2 \
    --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b {batch_size} -lr {lr} -rtg \
    --exp_name q2_b{batch_size}_r{lr}\
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
    run_result = subprocess.run(cmd.format(**hp_vals), shell=True, stdout=subprocess.PIPE , stderr=subprocess.PIPE)
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
        'lr': trial.suggest_int('lr', 1+1e-4, 1+0.1, log=True)-1,
        'batch_size': trial.suggest_int('batch_size', 1000, 25000, log=True),
    }
    avg_r = get_run_results(CMD_2, hp_vals)['Eval_AverageReturn']
    print("Real return:", avg_r)

    avg_r += math.log(hp_vals['lr'])
    avg_r -= math.log(hp_vals['batch_size']) / 2
    return avg_r


def main():
    # objective = optim_objective_1
    objective = optim_objective_2
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print()
    print("BEST PARAMS")
    print(study.best_params)

if __name__ == "__main__":
    main()
