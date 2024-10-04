#!/bin/bash

#SBATCH --job-name='cycleGAN'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -p general             
#SBATCH -q public
            
#SBATCH -t 1-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base

for resample in 2
do
    for generator_width in 160 
    do
        for num_residual_blocks_generator in 4 5
        do
            for discriminator_width in 150 160
            do
                for num_residual_blocks_discriminator in 5 6
                do
                    for lambda_cyc in 10
                    do
                        for lambda_id in 5
                        do
                            for lambda_mc in 0 1 3 5
                            do
                                patch_size=85
                                num_patch=1

                                batch_size=50
                                pool_size=2500

                                ~/.conda/envs/torch_base/bin/python main.py --resample $resample --generator_width $generator_width --num_residual_blocks_generator $num_residual_blocks_generator --discriminator_width $discriminator_width --num_residual_blocks_discriminator $num_residual_blocks_discriminator --lambda_cyc $lambda_cyc --lambda_id $lambda_id --lambda_mc $lambda_mc --log_path ./logs_newLS_${resample}_batch_${batch_size}_pool_${pool_size}_patch_${patch_size}_${num_patch}_long --lr 0.0002 --decay_epoch 100 --n_epochs 700  --sample_interval 10 --batch_size $batch_size --pool_size $pool_size --patch_size $patch_size --num_patch $num_patch --SUVR 1
                            done
                        done
                    done
                done
            done
        done
    done
done