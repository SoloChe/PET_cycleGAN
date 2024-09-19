#!/bin/bash

#SBATCH --job-name='cycleGAN'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -p general             
#SBATCH -q public
            
#SBATCH -t 1-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base

for resample in 1
do
    for generator_width in 128
    do
        for num_residual_blocks_generator in 6
        do
            for discriminator_width in 128
            do
                for num_residual_blocks_discriminator in 8
                do
                    for lambda_cyc in 4 6 8 10 12 14
                    do
                        for lambda_id in 3 5 7 9
                        do
                            ~/.conda/envs/torch_base/bin/python main.py --resample $resample --generator_width $generator_width --num_residual_blocks_generator $num_residual_blocks_generator --discriminator_width $discriminator_width --num_residual_blocks_discriminator $num_residual_blocks_discriminator --lambda_cyc $lambda_cyc --lambda_id $lambda_id --log_path ./training_logs_matching_para
                        done
                    done
                done
            done
        done
    done
done
