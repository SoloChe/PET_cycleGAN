#!/bin/bash

#SBATCH --job-name='baseline'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH -p general             
#SBATCH -q public
            
#SBATCH -t 0-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base

for resample in 1
do
    for generator_width in 128
    do
        for num_residual_blocks_generator in 4 6 8
        do
            for discriminator_width in 128
            do
                for num_residual_blocks_discriminator in 4 6 8 10
                do
                    for lambda_cyc in 10
                    do
                        for lambda_mc in 5
                        do
                            for lambda_id in 0
                            do
                                ~/.conda/envs/torch_base/bin/python main.py --resample $resample --generator_width $generator_width --num_residual_blocks_generator $num_residual_blocks_generator --discriminator_width $discriminator_width --num_residual_blocks_discriminator $num_residual_blocks_discriminator --lambda_cyc $lambda_cyc --lambda_id $lambda_id --lambda_mc $lambda_mc --log_path ./training_logs_matching_baseline --lr 0.0002 --decay_epoch 100 --n_epochs 500 --baseline 1
                            done
                        done
                    done
                done
            done
        done
    done
done