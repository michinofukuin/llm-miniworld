python train_freeway_rm.py --use_transfered_reward --seed 0 --gpu_id 0 &
python train_freeway_rm.py --seed 0 --gpu_id 1
python train_freeway_rm.py --use_transfered_reward --seed 1 --gpu_id 0 &
python train_freeway_rm.py --seed 1 --gpu_id 1
python train_freeway_rm.py --use_transfered_reward --seed 2 --gpu_id 0 &
python train_freeway_rm.py --seed 2 --gpu_id 1
python train_freeway_rm.py --use_transfered_reward --seed 3 --gpu_id 0 &
python train_freeway_rm.py --seed 3 --gpu_id 1
python train_freeway_rm.py --use_transfered_reward --seed 4 --gpu_id 0 &
python train_freeway_rm.py --seed 4 --gpu_id 1
python train_freeway_rm.py --use_transfered_reward --seed 5 --gpu_id 0 &
python train_freeway_rm.py --seed 5 --gpu_id 1
python train_freeway_rm.py --use_transfered_reward --seed 6 --gpu_id 0 &
python train_freeway_rm.py --seed 6 --gpu_id 1