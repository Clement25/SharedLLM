# SharedLLM

## Introduction
This repository contains the core code of the paper "Two Are Better than One: Context Window Extension with Multi-Grained Self-Injection".

## Usage

### Data Preparation
To train the model on downsampled redpajama and activation beacon, please refer to the following repositories to prepare the data

Downsampled RP: [https://github.com/princeton-nlp/CEPE](https://github.com/princeton-nlp/CEPE)

Activation-Beacon: [https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon)

### Training
To train sharedllm on red-pajama, use the following command to start training (by default we use NVIDIA 8xA800 GPUs for both training)
```bash
CUDA_VISIBLE_DEVICES=$CVD deepspeed train.py --model_name_or_path <llama_path> \
                                             --encoder_name_or_path <llama_path> \
                                             --config <path_to_config> \
`                                            --model_class sharedllm \
                                             --output_dir output/sharedllm_7b \
                                             --deepspeed <path_to_deepspeed_config>
```
For mixed dataset training, just change `train.py` to `train_beacon.py` and corresponding configuration file.

### Testing
For evaluation on language modeling, here's the example for testing model on 8K text length and arxiv domain. Here we use one A800 (80G) GPU to run this experiment
```bash
python eval_lm.py --config configs/test/test_ab_4x1024_4096 \
                  --model_name_or_path <path_to_model_ckpt> \
                  --model_class sharedllm  \
                  --validation_domains arxiv \
                  --output_dir output/<experiment_name>
```
For evaluation on longbench and Infbench, please refer to their respective repository and insert the model loading code to original evaluation scripts. Note, the input loader should be overwritten as original implementation supports decoder-only architectures (GPT) which differ from our implementation.

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
If you have any further question about the code, feel free to contact me via henryhan88888@gmail.com.
