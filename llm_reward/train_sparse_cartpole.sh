python train_sparse_cartpole.py --use_reward_machine --use_transfered_reward --seed 0 --gpu_id 1 &
python train_sparse_cartpole.py --use_transfered_reward --seed 0 --gpu_id 2 &
python train_sparse_cartpole.py --use_reward_machine --seed 0 --gpu_id 3 &
python train_sparse_cartpole.py --seed 0 --gpu_id 4;
python train_sparse_cartpole.py --use_reward_machine --use_transfered_reward --seed 1 --gpu_id 1 &
python train_sparse_cartpole.py --use_transfered_reward --seed 1 --gpu_id 2 &
python train_sparse_cartpole.py --use_reward_machine --seed 1 --gpu_id 3 &
python train_sparse_cartpole.py --seed 1 --gpu_id 4;
python train_sparse_cartpole.py --use_reward_machine --use_transfered_reward --seed 2 --gpu_id 1 &
python train_sparse_cartpole.py --use_transfered_reward --seed 2 --gpu_id 2 &
python train_sparse_cartpole.py --use_reward_machine --seed 2 --gpu_id 3 &
python train_sparse_cartpole.py --seed 2 --gpu_id 4;