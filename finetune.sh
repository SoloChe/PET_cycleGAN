#!/bin/bash

#SBATCH --job-name='finetune'
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -p general             
#SBATCH -q public
            
#SBATCH -t 0-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate torch_base

# 0.9767; 85 1 10 500 0.00009 ft1 with 17 1 10
# folder=logs_newnew_2_batch_100_pool_5000_patch_85_1_long
# setup=160_150_4_4_10.0_5.0_0.0
# SUVR=1

# 0.9807 0.00011 15 0 0 seed=30
# folder=logs_newLS_2_batch_50_pool_2500_patch_85_1_long
# setup=180_200_4_5_10.0_5.0_0.0
# SUVR=1

# folder=logs_RSF_2_batch_50_pool_2500_patch_85_1_long
# setup=160_150_4_5_10.0_5.0_5.0
# SUVR=0

## original keep CL<=30
# folder=logs_CL_2_batch_50_pool_2500_patch_86_1_long
# setup=200_220_4_4_10.0_5.0_0.0
# SUVR=1
# max_cor=0.9823

## ft1 keep CL>30
# folder=logs_ftCL2_1_batch_1_pool_50_patch_86_1_long
# setup=200_220_4_4_10.0_1.0_0.0
# SUVR=1
# max_cor=0.9862


## ft2 keep CL<=30
# folder=logs_ftCL3_1_batch_1_pool_50_patch_86_1_long
# setup=200_220_4_4_10.0_1.0_0.0
# SUVR=1
# max_cor=0.9876

folder=logs_NCL_test_1_batch_50_pool_2500_patch_85_1_long
setup=200_220_4_5_10.0_5.0_0.0
SUVR=1
max_cor=0.9717

model_path=./$folder/saved_model/$setup
data_path=./$folder/data/$setup

for resample in 1
do
    for generator_width in 200
    do
        for num_residual_blocks_generator in 4
        do
            for discriminator_width in 220
            do
                for num_residual_blocks_discriminator in 5
                do
                    for lambda_cyc in 10
                    do
                        for lambda_id in 0
                        do
                            for lambda_mc in 0
                            do

                                patch_size=85
                                num_patch=1

                                batch_size=20
                                pool_size=100

                                # finetune_lr=0.00001
                                # finetune_lr=0.00002
                                finetune_lr=0.00004
                                # finetune_lr=0.00010
                                # finetune_lr=0.00012
                                # finetune_lr=0.00013

                                # seed=30
                                seed=30
                                ~/.conda/envs/torch_base/bin/python main.py --resample $resample --generator_width $generator_width --num_residual_blocks_generator $num_residual_blocks_generator --discriminator_width $discriminator_width --num_residual_blocks_discriminator $num_residual_blocks_discriminator --lambda_cyc $lambda_cyc --lambda_id $lambda_id --lambda_mc $lambda_mc --log_path ./logs_ftNCL_${resample}_batch_${batch_size}_pool_${pool_size}_patch_${patch_size}_${num_patch}_long  --decay_epoch 1 --n_epochs 100 --sample_interval 1 --batch_size $batch_size --pool_size $pool_size --patch_size $patch_size --num_patch $num_patch --model_path $model_path --finetune 1 --finetune_lr $finetune_lr --SUVR $SUVR --seed $seed --shuffle 1 --max_cor $max_cor --data_path $data_path
                            done
                        done
                    done
                done
            done
        done
    done
done