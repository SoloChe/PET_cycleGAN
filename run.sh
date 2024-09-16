#!/bin/bash

#SBATCH --job-name='cycleGAN'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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
    for generator_width in 1024
    do
        for num_residual_blocks_generator in 2 4 
        do
            for discriminator_width in 1024
            do
                for num_residual_blocks_discriminator in 2 4 6
                do
                    ~/.conda/envs/torch_base/bin/python cycleGAN.py --resample $resample --generator_width $generator_width --num_residual_blocks_generator $num_residual_blocks_generator --discriminator_width $discriminator_width --num_residual_blocks_discriminator $num_residual_blocks_discriminator
                done
            done
        done
    done
done
