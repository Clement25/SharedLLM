CVD=$1
NGPU=${#CVD[@]}

CUDA_VISIBLE_DEVICES=$CVD torchrun --nnodes=1 --nproc_per_node=$NGPU train.py --config configs/train_llama2_warmup