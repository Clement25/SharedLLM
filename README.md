# SharedLLM

## Introduction
This repository contains the core code of the paper "Two Are Better than One: Context Window Extension with Multi-Grained Self-Injection".

## Usage

### Data Preparation
To train the model on downsampled redpajama and activation beacon, please refer to the following repositories to prepare the data

Downsampled RP: [https://github.com/princeton-nlp/CEPE](https://github.com/princeton-nlp/CEPE)

Activation-Beacon: [https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon)

### Training
To train sharedllm on red-pajama, use the following command to start training
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train.py --model_name_or_path <llama_path> --encoder_name_or_path <llama_path> --config <path_to_config> --model_class sharedllm --output_dir output/sharedllm_7b --deepspeed <path_to_deepspeed_config>
```
For mixed dataset training, just change train.py to train_beacon.py and corresponding configuration file.

### Testing
For language modeling
```
python eval_lm.py --config configs/test_ab_4x1024_4096 --model_name_or_path <path_to_model_ckpt> --model_class sharedllm --validation_domains arxiv --output_dir output/<experiment_name>
```
For evaluation on longbench and Infbench, please refer to their respective repository and insert the model loading code to original evaluation scripts. Note, the input loader should be overwritten as original implementation supports decoder-only architectures (GPT) which differ from our implementation.

## Citation
If you have any further question about the code, feel free to contact me via henryhan88888@gmail.com.