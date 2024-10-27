import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "ceped-llama2-7b-chat", "ceped-llama2-7b-ft", 
    "td-llama2-7b-chat", "td-llama2-7b-tree", "td-llama2-7b-test1", "td-llama2-7b-test2", "td-llama2-7b-test3", "td-llama2-7b-test4", "td-llama2-7b-test5", "td-llama2-7b-test6", "td-llama2-7b-test7", "td-llama2-7b-test8",
    "llama3-8b-chat-8k", "longalpaca-16k", "td-llama2-lora-7b-chat"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama" in model_name :
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            # prompt = build_chat(tokenizer, prompt, model_name)
            prompt = prompt

        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        elif "ceped" in model_name:
            all_input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            all_length = all_input.input_ids.shape[-1]
            bsz = all_input.input_ids.shape[0]
            input = dict()
            # determine num_chunks, reorganize inputs 
            chunk_length = 256

            # loader v2
            num_encoder_chunk = min(max(all_length // chunk_length - 1, 1), 16)

            # loader v1
            # num_encoder_chunk = max((all_length - 3404) // chunk_length, 1)
            encoder_input_length = num_encoder_chunk * chunk_length
            input['input_ids'] = all_input.input_ids[...,encoder_input_length:].to(device)
            input['attention_mask'] = all_input.attention_mask[..., encoder_input_length:].to(device)
            input['encoder_input_ids'] = all_input.input_ids[...,:encoder_input_length].reshape(bsz, num_encoder_chunk, chunk_length).to(device)
            input['encoder_attention_mask'] = all_input.attention_mask[..., :encoder_input_length].reshape(bsz, num_encoder_chunk, chunk_length).to(device)

        elif "td-llama2" in model_name:
            all_input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            bsz, all_length = all_input.input_ids.shape

            input = dict()
            # determine num_chunks, reorganize inputs 
            # chunk_length = 1024 if '1024x' in model2path[model_name] else 512
            chunk_length = 512

            import math
            TRUNC_LEN = {
                "td-llama2-7b-chat": 3584,
                "td-llama2-7b-tree": 3072,
                "td-llama2-7b-test1": 2560,
                "td-llama2-7b-test2": 2304,
                "td-llama2-7b-test3": 2048,
                "td-llama2-7b-test4": 1792,
                "td-llama2-7b-test5": 1536,
                "td-llama2-7b-test6": 1280,
                "td-llama2-7b-test7": 1024,
                "td-llama2-7b-test8": 512
                # "td-llama2-lora-7b-chat": 2048
            }[model_name]
            num_encoder_chunk = max(math.ceil((all_length - TRUNC_LEN) // chunk_length), 1)

            # works better on Beacon
            # num_encoder_chunk = min(max(all_length // chunk_length - 1, 1), 8)

            if "presuf" in model2path[model_name]:
                PREFIX_SIZE = 1
                SUFFIX_SIZE = 64
                input["encoder_prefix_id"] = all_input.input_ids[..., :PREFIX_SIZE].to(device)
                input["encoder_suffix_id"] = all_input.input_ids[..., -SUFFIX_SIZE:].to(device)
            else:
                PREFIX_SIZE = 0
                SUFFIX_SIZE = 0

            if all_length < chunk_length:   # too short context, only to continual
                continue

            encoder_input_length = num_encoder_chunk * chunk_length
            encoder_input_ids = all_input.input_ids[..., PREFIX_SIZE:PREFIX_SIZE+encoder_input_length]
            encoder_attention_mask = all_input.attention_mask[..., PREFIX_SIZE:PREFIX_SIZE+encoder_input_length]

            input['encoder_input_ids'] = encoder_input_ids.reshape(bsz, num_encoder_chunk, chunk_length).to(device)
            input['encoder_attention_mask'] = encoder_attention_mask.reshape(bsz, num_encoder_chunk, chunk_length).to(device)

            ## if input is too short (i.e., just fill up encoder 1 chunk and no left for decoder, we fill decoder with the same input)
            input['input_ids'] = all_input.input_ids[..., PREFIX_SIZE+encoder_input_length:].to(device)
            input['attention_mask'] = all_input.attention_mask[..., PREFIX_SIZE+encoder_input_length:].to(device)

        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input['input_ids'].shape[-1]

        # generation configs
        if 'longalpaca' in model_name:
            top_p = 0.9
            temperature = 0.6
        else:
            top_p, temperature = 1.0, 1.0

        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
        # if dataset in ["2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper", "narrativeqa", "samsum"]:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=temperature,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=temperature,
            )[0]


        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f: 
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    # dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "ceped-llama2" in model_name:  # for ceped
        from modeling.modeling_llama_flash import LlamaForCausalContextLM
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalContextLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "td-llama2" in model_name: # for td llama
        if 'lora' in model_name: # lora version
            from modeling.modeling_llama_tdret_lora_flash import TopDownLlamaForCausalLMv4LoRAFTMulInst, TopDownLlamaForCausalLMv5LoRAFTMulInst, TopDownLlamaForCausalLMv6LoRAFTMulInst
            tokenizer = LlamaTokenizer.from_pretrained(path)
            if 'enclora' in path:
                if 'presuf' in path:
                    model = TopDownLlamaForCausalLMv5LoRAFTMulInst.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
                else:
                    model = TopDownLlamaForCausalLMv4LoRAFTMulInst.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
            elif 'fulllora' in path:
                model = TopDownLlamaForCausalLMv6LoRAFTMulInst.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
            else:
                raise ValueError('undefined model type!')
        else:   # full FT version
            from modeling.modeling_llama_tdret_flash import  TopDownLlamaForCausalLMv4FTMulInst, TopDownLlamaForCausalLMv5FTMulInst
            tokenizer = LlamaTokenizer.from_pretrained(path)
            if 'presuf' in path:
                model = TopDownLlamaForCausalLMv5FTMulInst.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
            else:
                model = TopDownLlamaForCausalLMv4FTMulInst.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)

    elif "llama2" in model_name or "llama3" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longalpaca" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    model = model.eval()
    return model, tokenizer

def main():
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # 21 tasks
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        # English tasks (16 tasks)
        datasets = [
                    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum","multi_news",
                    "trec", "triviaqa", "samsum", \
                    "lcc", "repobench-p"
                ]
        # Synthetic tasks (2 tasks)
        # datasets = ["passage_count", "passage_retrieval_en"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        print(f'Now evaluating on dataset {dataset}...')
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

def main_debug():
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    # mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # 21 tasks
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        # English tasks
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "trec", "triviaqa", "samsum", \
                    "passage_count", "passage_retrieval_en", "lcc", "repobench-p", "gov_report", "qmsum","multi_news"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        print(f'Now evaluating on dataset {dataset}...')
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        for rank in range(world_size):
            get_pred(rank, world_size, data_subsets[rank], max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path)

if __name__ == '__main__':
    main()
    # main_debug()
