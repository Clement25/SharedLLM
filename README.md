# SharedLLM

## Introduction
This repository contains the core code of the paper "[Stacked from One: Multi-Scale Self-Injection for Context Window Extension](https://arxiv.org/pdf/2410.19318)", accepted by ICLR 2026.

## Usage

### Data Preparation
To train the model on downsampled redpajama and activation beacon, please refer to the following repositories to prepare the data

1. Pretraining
Downsampled RP: [https://github.com/princeton-nlp/CEPE](https://github.com/princeton-nlp/CEPE)

2. SFT
LongAlpaca-12K: [https://huggingface.co/datasets/Yukang/LongAlpaca-12k](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)
BookSum: [https://huggingface.co/datasets/kmfoda/booksum](https://huggingface.co/datasets/kmfoda/booksum)

Synthetic Data (highly encouraged): See details in [Llama-3-8B-Instruct-262k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k) and [Synthetic Data for Multi-Doc QA](https://arxiv.org/pdf/2401.03462)

### Training
To train sharedllm on red-pajama, use the following command to start training (by default we use NVIDIA 8xA800 GPUs with deepspeed)
```bash
CUDA_VISIBLE_DEVICES=$CVD deepspeed train.py --model_name_or_path <llama_path> \
                                             --encoder_name_or_path <llama_path> \
                                             --config <path_to_config> \
                                             --model_class sharedllm \
                                             --output_dir output/sharedllm_7b \
                                             --deepspeed <path_to_deepspeed_config>
```
For mixed dataset training, just change `train.py` to `train_beacon.py` and the corresponding configuration files.

### Testing
For evaluation on language modeling, here's an example of testing model on 8K text length and arxiv domain. Here we use one A800 (80G) GPU to run this experiment
```bash
python eval_lm.py --config configs/test/test_ab_4x1024_4096 \
                  --model_name_or_path <path_to_model_ckpt> \
                  --model_class sharedllm  \
                  --validation_domains arxiv \
                  --output_dir output/<experiment_name>
```
For evaluation on longbench and Infbench, please refer to their respective repository and insert the model loading code to the original evaluation scripts. Note that the input loader should be modified as original implementation only supports decoder-only architectures (GPT) which differs from ours.

## Citation
```bibtex
@misc{han2024betteronecontextwindow,
      title={Two are better than one: Context window extension with multi-grained self-injection}, 
      author={Wei Han and Pan Zhou and Soujanya Poria and Shuicheng Yan},
      year={2024},
      eprint={2410.19318},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.19318}, 
}
```
If you have any further questions about this work, feel free to contact me via henryhan88888@gmail.com.
