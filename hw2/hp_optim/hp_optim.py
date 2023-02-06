import subprocess
import json
import optuna

CMD_1 = "\
python run_hw2.py --env_name CartPole-v0 -n 100 --data_path data_tmp \
-b {batch_size} {dsa} {rtg} --exp_name q1_sb_no_rtg_dsa \
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
            print(parts[0], ':', parts[1])
            data[parts[0]] = float(parts[1])
    return data

def optim_objective_1(trial):
    hp_vals = {
        'dsa': trial.suggest_categorical('dsa', ['', '-dsa']),
        'rtg': trial.suggest_categorical('rtg', ['', '-rtg']),
        'batch_size': trial.suggest_int('batch_size', 100, 20000),
    }
    return get_run_results(CMD_1, hp_vals)['Eval_AverageReturn']


def main():
    objective = optim_objective_1
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print()
    print("BEST PARAMS")
    print(study.best_params)

if __name__ == "__main__":
    main()
