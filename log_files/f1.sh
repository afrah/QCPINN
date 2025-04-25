#!/bin/bash
#SBATCH -J  "qnn4pde"
#SBATCH -A  kddcik
#SBATCH -p  core40q
#SBATCH -o  h_%j.out
#SBATCH -e  h_%j.err
#SBATCH -N  1
#SBATCH -n  1

## SBATCH --nodelist=s030  # Added to specify the idle nodes

##source /okyanus/progs/ANACONDA/anaconda3-2020.07-python3.8.3/etc/profile.d/conda.sh
source ~/.bashrc

export CUDA_VISIBLE_DEVICES=-1


conda activate pennylane_env
echo "date :"$date

pwd
export PYTHONPATH="${PYTHONPATH}:/okyanus/users/afarea/afrah/code/QNN4PDEs"

# python -m src.testing.cavity_test
# python -m src.testing.helmholtz_test
# python -m src.testing.helmholtz_test

python -m src.poisson.cavity_hybrid_trainer
# python -m src.poisson.wave_hybrid_trainer
# python -m src.poisson.klein_gordon_hybrid_trainer
# python -m src.poisson.diffusion_hybrid_trainer
# python -m src.poisson.helmholtz_hybrid_trainer

# python -m src.contour_plots.cavity_hybrid_plotting
# python -m src.contour_plots.helmholtz_hybrid_plotting

# python -m src.contour_plots.wave_hybrid_plotting
# python -m src.contour_plots.klein_gordon_hybrid_plotting
# python -m src.contour_plots.diffusion_hybrid_plotting



echo ""
if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  echo "print =========================================="
squeue  --job $SLURM_JOB_ID

echo "print SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "print SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "print SLURM_SUBMIT_HOST = $SLURM_SUBMIT_HOST"
  echo "print SLURM_SUBMIT_DIR = $SLURM_SUBMIT_DIR"
  echo "print SLURM_JOB_NUM_NODES = $SLURM_JOB_NUM_NODES"
  echo "print SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
  echo "print SLURM_NTASKS = $SLURM_NTASKS"
  echo "print SLURM_NODEID = $SLURM_NODEID"
echo "print =========================================="
fi
