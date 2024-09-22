#!/bin/bash

#SBATCH --job-name='cycleGAN_sep'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -p general             
#SBATCH -q public
            
#SBATCH -t 1-00:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base

for resample in 1
do
    for generator_width in 256
    do
        for num_residual_blocks_generator in 4
        do
            for discriminator_width in 256
            do
                for num_residual_blocks_discriminator in 10
                do
                    for lambda_cyc in 10
                    do
                        for lambda_mc in 0
                        do
                            for lambda_id in 3 7
                            do
                                ~/.conda/envs/torch_base/bin/python main_sep.py --resample $resample --generator_width $generator_width --num_residual_blocks_generator $num_residual_blocks_generator --discriminator_width $discriminator_width --num_residual_blocks_discriminator $num_residual_blocks_discriminator --lambda_cyc $lambda_cyc --lambda_id $lambda_id --lambda_mc $lambda_mc --log_path ./training_logs_matching_sep --lr 0.0002 --decay_epoch 100 --n_epochs 500
                            done
                        done
                    done
                done
            done
        done
    done
done