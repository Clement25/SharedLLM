CVD=$1
NGPU=${#CVD[@]}

CUDA_VISIBLE_DEVICES=$CVD python eval_lm.py --config myconfigs/test_ab_32k_prevdoc_28x1024_2304 --model_name_or_path /mnt/data/weihan/models/Llama-2-7b-hf --model_class topdown --validation_domains book --output_dir output/Hier-LLaMA-7b --encoder_path /mnt/data/weihan/models/Llama-2-7b-chat-hf