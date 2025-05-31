#!/bin/bash

#SBATCH --job-name='cycleGAN'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH -p general             
#SBATCH -q public
            
#SBATCH -t 0-05:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base


for generator_width in 300 
do
    for num_residual_blocks_generator in 4 
    do
        for discriminator_width in 300 
        do
            for num_residual_blocks_discriminator in 4 
            do
                for lambda_cyc in 10 
                do
                    for lambda_id in 5
                    do
                        for lambda_mc in 0
                        do
                            for seed in 30
                            do
                                for train_size in  11000
                                do

                                # batch size and pool size for buffer
                                batch_size=20
                                pool_size=500

                                # patch discriminator
                                patch_size=85 #85: no cl and dm 86: with cl 88: with cl and dm
                                num_patch=1


                                ~/.conda/envs/torch_base/bin/python main_test.py --generator_width $generator_width --num_residual_blocks_generator $num_residual_blocks_generator --discriminator_width $discriminator_width --num_residual_blocks_discriminator $num_residual_blocks_discriminator --lambda_cyc $lambda_cyc --lambda_id $lambda_id --lambda_mc $lambda_mc --log_path ./logs_scale6/logs_${train_size}_batch_${batch_size}_pool_${pool_size}_patch_${patch_size}_${num_patch}_${seed} --lr 0.0002 --decay_epoch 100 --n_epochs 700  --sample_interval 10 --batch_size $batch_size --pool_size $pool_size --patch_size $patch_size --num_patch $num_patch --seed $seed --train_size $train_size 
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done