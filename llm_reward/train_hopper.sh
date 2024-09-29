python train_hopper.py --use_reward_machine --use_transfered_reward --seed 0 --gpu_id 1 &
python train_hopper.py --use_transfered_reward --seed 0 --gpu_id 2 &
python train_hopper.py --use_reward_machine --seed 0 --gpu_id 3 &
python train_hopper.py --seed 0 --gpu_id 4 &
python train_hopper.py --use_reward_machine --use_transfered_reward --seed 1 --gpu_id 5 &
python train_hopper.py --use_transfered_reward --seed 1 --gpu_id 6 &
python train_hopper.py --use_reward_machine --seed 1 --gpu_id 7 &
python train_hopper.py --seed 1 --gpu_id 8;
python train_hopper.py --use_reward_machine --use_transfered_reward --seed 2 --gpu_id 1 &
python train_hopper.py --use_transfered_reward --seed 2 --gpu_id 2 &
python train_hopper.py --use_reward_machine --seed 2 --gpu_id 3 &
python train_hopper.py --seed 2 --gpu_id 4;