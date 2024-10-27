CVD=$1
NGPU=${#CVD[@]}

# 8K (1GPU)
CUDA_VISIBLE_DEVICES=${CVD[0]} python eval_lm.py --config myconfigs/test_proofpile_4x1024_3840 --model_name_or_path /mnt/data/weihan/models/Llama-2-7b-hf --model_class topdown --validation_domains proofpile --output_dir output/Hier-LLaMA-7b --encoder_path /mnt/data/weihan/models/Llama-2-7b-chat-hf

# 32K (2GPU)
CUDA_VISIBLE_DEVICES=$CVD python eval_lm.py --config myconfigs/test_proofpile_32k_28x1024_2304 --model_name_or_path /mnt/data/weihan/models/Llama-2-7b-hf --model_class topdown --validation_domains proofpile --output_dir output/Hier-LLaMA-7b --encoder_path /mnt/data/weihan/models/Llama-2-7b-chat-hf