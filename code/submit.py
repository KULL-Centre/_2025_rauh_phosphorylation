import subprocess
import os
import pandas as pd
import numpy as np
import mdtraj as md
import shutil
import time
from jinja2 import Template

submission = Template("""#!/bin/sh
#SBATCH --job-name={{folder}}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=sbinlab_ib2
#SBATCH --mem=2GB
#SBATCH -t 20:00:00
#SBATCH -o {{folder}}/out
#SBATCH -e {{folder}}/err

source /groups/sbinlab/asrauh/.bashrc

conda activate phospho


echo $SLURM_CPUS_PER_TASK

echo $SLURM_CPUS_ON_NODE

#SCRATCH=/scratch/$USER/$SLURM_JOBID
#mkdir -p $SCRATCH && cd $SCRATCH

hostname

python ./simulate.py --name {{name}} --folder {{folder}} --dlambda {{dlambda}}

#mv $SCRATCH/* {{folder}}""")

proteins = pd.read_csv('proteins.csv', index_col=0)

if not os.path.exists('/groups/sbinlab/asrauh/software/BLOCKING'):
    subprocess.check_call(['git', 'clone', 'https://github.com/fpesceKU/BLOCKING', './BLOCKING'])

delta_lambdas = np.arange(-.62,0,.05) # np.arange(-0.37,0,0.05) -1.02
# or lambdas = np.arange(0,0.3,0.05)
num_replicas = 3  # Define the number of replicas

# or for lam in lambdas:
for delta_lambda in delta_lambdas:
    folder = f'./{delta_lambda:.2f}'
    delta_lambda = np.round(delta_lambda,2)
    os.makedirs(folder, exist_ok=True)

    residues = pd.read_csv('residues.csv', index_col=0)
    residues.loc['B', 'lambdas'] += delta_lambda
    residues.loc['O', 'lambdas'] += delta_lambda
    residues.loc['U', 'lambdas'] += delta_lambda
    # or residues.loc['B', 'lambdas'] = lam
    # residues.loc['O', 'lambdas'] = lam

    residues.to_csv(folder+'/residues.csv')

    for name in ['pERa']:# proteins.index[:]: 'pTauS','pTauT'
        for k in range(num_replicas):
            subfolder = folder+F'/{name:s}/{k:d}'
            if not os.path.isdir(subfolder):
                os.makedirs(subfolder, exist_ok=True)
            with open(f'{subfolder:s}/submit.sh', 'w') as submit:
                submit.write(submission.render(name=name, folder=subfolder,dlambda=delta_lambda))           
            print(F"{delta_lambda}-{name}-{k}")
            subprocess.run(['sbatch', f'{subfolder:s}/submit.sh'])
            time.sleep(0.6)
