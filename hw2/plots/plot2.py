from tbparse import SummaryReader
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_path = Path('data_questions/data_q5')
run_returns = {}
for run_path in data_path.iterdir():
    if not 'Hopper' in run_path.name:
        continue
    name_parts = run_path.name.split('_')
    lbda = float(name_parts[5][len('lambda'):])
    run_data = SummaryReader(str(run_path)).scalars
    avg_return = run_data[run_data['tag'] == 'Eval_AverageReturn']['value'].to_numpy()
    run_returns[lbda] = avg_return
lbda_values = sorted(run_returns.keys())

# ==================================================

allowed_values = set([0, 0.95, 0.99, 1])

X = np.arange(len(run_returns[lbda_values[0]]))+1

for lbda in lbda_values:
    if lbda in allowed_values:
        plt.plot(X, run_returns[lbda], label=f'$\lambda$={lbda}')

plt.xlabel('Number of epochs')
plt.ylabel('Average return')

plt.ylim(ymin=0)
plt.legend(loc='upper left')
plt.show()

# ==================================================

plt.plot(lbda_values, [run_returns[l][-1] for l in lbda_values])
plt.xlabel('$\lambda$')
plt.ylabel('Final average return')
plt.ylim(ymin=0)
plt.show()