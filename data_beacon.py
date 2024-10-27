import datasets
import numpy as np
import torch
from dataclasses import dataclass
from glob import glob
from collections import defaultdict, OrderedDict
from typing import Optional, Tuple, Union, List, Callable, Dict, Any, Mapping
from transformers.tokenization_utils import PreTrainedTokenizer

def are_elements_of_same_length(lst: List):
    if not isinstance(lst[0], list):
        return False

    length = len(lst[0])
    return all(len(x) == length if isinstance(x, list) else False for x in lst)

def add_eos(inputs: Mapping, eos_token_id: int):
    """Add eos for BatchEncoding object."""
    assert isinstance(inputs["input_ids"], list), f"Make sure the return_tensors are set to list!"
    if inputs["input_ids"][-1] != eos_token_id:
        for k, v in inputs.items():
            if k in ["input_ids", "labels"]:
                v = v + [eos_token_id]
            elif k == "position_ids":
                v = v + [v[-1] + 1]
            elif k in ["attention_mask", "token_type_ids"]:
                v = v + v[-1:]
            else:
                raise NotImplementedError(f"Inputs key {k} not implemented!")
            inputs[k] = v
    return inputs

class DatasetProcessFn:
    """Wrapper for any user-defined process function for huggingface datasets.

    1. Process batched examples by looping the process function over them;
    2. Gather returned examples if any data augmentation happens with augment=True;
    3. Pass indices of examples inside the process function with _index keywords if they exist.

    The wrapped function should take in any needed columns and return a dict with 1 or more samples.
    """
    def __init__(self, augment=False):
        self.augment = augment


    def __call__(self, _process_fn):
        def process(*args):
            sample_or_batch_sample = args[0]
            if len(args) == 1:
                pass
            elif len(args) == 2:
                indices = args[1]
                # detach the slice so that _index will not be set in the original data
                sample_or_batch_sample = sample_or_batch_sample.copy()
                sample_or_batch_sample["_index"] = indices
            else:
                raise NotImplementedError(f"Found more than 2 arguments {args}!")


            keys = list(sample_or_batch_sample.keys())
            func_args = [sample_or_batch_sample[k] for k in keys]
            # FIXME: if all values in one sample are of the same length, this would fail
            if are_elements_of_same_length(func_args):
                outputs = defaultdict(list)
                for arg in zip(*func_args):
                    # get each element in a batch
                    kwargs = {keys[j]: arg[j] for j in range(len(arg))}
                    output = _process_fn(**kwargs)
                    if output is not None:
                        for k, v in output.items():
                            if self.augment:
                                outputs[k].extend(v)
                            else:
                                outputs[k].append(v)
            else:
                outputs = _process_fn(**sample_or_batch_sample)
                if outputs is None:
                    raise ValueError(f"Found None returned from process_fn. Make sure you set 'batched=True' when trying to augment/distract samples in the datasets!")
            return dict(outputs)
        return process


class Data:
    @staticmethod
    def get_process_fn(tokenizer, min_length, max_length, context_size, seed=42, with_labels=True):
        patterns = [" ", "\n", "\n\n"]
        rng = np.random.default_rng(seed=seed)
        
        @DatasetProcessFn()
        def process_fn(text=None, input=None, output=None, input_ids=None, labels=None, _index=None, **kwds):

            CHUNK_SIZE = context_size
            inputs = dict()
            if text is not None:
                # truncate text for faster processing
                text = text[:max_length * 5]
                if with_labels:
                    all_inputs = tokenizer(text, truncation=True, max_length=max_length)
                    if len(all_inputs["input_ids"]) < min_length:
                        return None

                    MAX_CHUNK = 8
                    MAX_NUM_CHUNK = min((8192 - 1024) // CHUNK_SIZE, MAX_CHUNK)

                    num_encoder_chunks = min(max(len(all_inputs.input_ids) // CHUNK_SIZE - 1, 1), MAX_NUM_CHUNK)
                    encoder_lengths = num_encoder_chunks * CHUNK_SIZE

                    inputs['encoder_input_ids'] = np.array(all_inputs.input_ids[:encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()
                    inputs['encoder_attention_mask'] = np.array(all_inputs.attention_mask[:encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()

                    inputs['input_ids'] = all_inputs['input_ids'][encoder_lengths:]
                    inputs['attention_mask'] = all_inputs['attention_mask'][encoder_lengths:]
                    #### Add-on ends

                    labels = inputs["input_ids"].copy()
                    inputs["labels"] = labels
                    
            elif input is not None:
                input = input.strip()
                output = output.strip() + tokenizer.eos_token
                # for too long inputs, we truncate first to save encoding time
                if len(input) > 5 * max_length:
                    input = input[:(5 * max_length) // 2] + input[-(5 * max_length) // 2:]

                tokenized_input = tokenizer.encode(input)
                tokenized_output = tokenizer.encode(output, add_special_tokens=False)
                output_length = len(tokenized_output)                
                # some outputs are too long, discard them
                if output_length > max_length:
                    return None

                # truncate from middle
                input_max_length = max_length - output_length
                if len(tokenized_input) > input_max_length:
                    half = int(input_max_length / 2)
                    input = tokenizer.decode(tokenized_input[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_input[-half:], skip_special_tokens=True)

                if with_labels:
                    pattern = rng.choice(patterns).tolist()
                    all_inputs = tokenizer(pattern.join([input, output]))
                    maybe_truncated_input=tokenizer.encode(input)
                    if len(all_inputs["input_ids"]) < min_length:
                        return None

                    #### split into encoder and decoder part
                    num_encoder_chunks = max(len(maybe_truncated_input) // CHUNK_SIZE - 1, 1)
                    encoder_lengths = num_encoder_chunks * CHUNK_SIZE

                    ### determine encoder input
                    inputs['encoder_input_ids'] = np.array(all_inputs.input_ids[:encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()
                    inputs['encoder_attention_mask'] = np.array(all_inputs.attention_mask[:encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()

                    inputs['input_ids'] = all_inputs['input_ids'][encoder_lengths:]
                    inputs['attention_mask'] = all_inputs['attention_mask'][encoder_lengths:]
                    #### Add-on ends

                    ### determine decoder input
                    labels = inputs["input_ids"].copy()
                    labels[:-output_length] = [-100 for _ in range(len(labels) - output_length)]
                    inputs["labels"] = labels
                else:
                    inputs = tokenizer(input)
                    raise ValueError('with labels must be true')

            elif input_ids is not None:
                # if len(input_ids) < min_length or len(input_ids) > max_length:
                #     return None
                # inputs = {
                #     "input_ids": input_ids,
                #     "labels": labels,
                # }
                raise NotImplementedError(f"Direct processing input_ids is not implemented")

            else:
                raise NotImplementedError(f"The dataset must contain one of the following fields: 'input' & 'output' (for fine-tuning), 'text' (for pre-training), 'input_ids' & 'labels' for passkey_retrieval.")

            # index is required for evaluation, by default we always add it
            inputs["index"] = _index
            # length is required for grouping
            inputs["length"] = len(inputs["input_ids"])
            return inputs
        return process_fn

    @staticmethod
    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, context_size=512, max_train_num_per_data=None, seed=42, cache_dir=None):
        if data_files is None:
            return None
        if isinstance(data_files, str):
            if "*" in data_files:
                data_files = glob(data_files)
            else:
                data_files = [data_files]
        
        process_fn = Data.get_process_fn(tokenizer, max_length=max_length, min_length=min_length, context_size=context_size, seed=seed)

        train_datasets = []
        for path in data_files:
            temp_dataset = datasets.load_dataset('json', data_files=path, split='train', cache_dir=cache_dir)
            temp_dataset = temp_dataset.map(process_fn, batched=True, num_proc=96, batch_size=1280, remove_columns=temp_dataset.column_names)
            if max_train_num_per_data is not None and len(temp_dataset) > max_train_num_per_data:
                temp_dataset = temp_dataset.train_test_split(max_train_num_per_data, seed=seed)["test"]
            train_datasets.append(temp_dataset)

        dataset = datasets.concatenate_datasets(train_datasets)
        return dataset


class DataMixed:
    @staticmethod
    def get_process_fn(tokenizer, min_length, max_length, context_size, seed=42, with_labels=True, thr=0.0):
        patterns = [" ", "\n", "\n\n"]
        rng = np.random.default_rng(seed=seed)
        
        @DatasetProcessFn()
        def process_fn(text=None, input=None, output=None, input_ids=None, labels=None, _index=None, **kwds):

            CHUNK_SIZE = context_size
            inputs = dict()
            if text is not None:
                # truncate text for faster processing
                text = text[:max_length * 5]
                if with_labels:
                    all_inputs = tokenizer(text, truncation=True, max_length=max_length)
                    if len(all_inputs["input_ids"]) < min_length:
                        return None

                    MAX_CHUNK = 8
                    TRUNC = 1024
                    MAX_NUM_CHUNK = min((8192 - TRUNC) // CHUNK_SIZE, MAX_CHUNK)

                    import random
                    # continuation modeling
                    if random.random() > thr:   
                        num_encoder_chunks = min(max(len(all_inputs.input_ids) // CHUNK_SIZE - 1, 1), MAX_NUM_CHUNK)
                        encoder_lengths = num_encoder_chunks * CHUNK_SIZE

                        inputs['encoder_input_ids'] = np.array(all_inputs.input_ids[:encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()
                        inputs['encoder_attention_mask'] = np.array(all_inputs.attention_mask[:encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()

                        inputs['input_ids'] = all_inputs['input_ids'][encoder_lengths:]
                        inputs['attention_mask'] = all_inputs['attention_mask'][encoder_lengths:]
                        #### Add-on ends
                        labels = inputs["input_ids"].copy()
                        inputs["labels"] = labels
                    # create copy task
                    else:   
                        COPY_LEN = 2048
                        PRED_LEN = COPY_LEN // 1
                        APPEND_LEN = 0

                        REAL_CHUNK_SIZE = CHUNK_SIZE - APPEND_LEN
                        from math import ceil
                        # num_encoder_chunks = ceil(len(all_inputs.input_ids) / CHUNK_SIZE)
                        num_encoder_chunks = ceil(len(all_inputs.input_ids) / REAL_CHUNK_SIZE)
                        encoder_lengths = num_encoder_chunks * REAL_CHUNK_SIZE

                        # placeholders of input_ids
                        encoder_input_ids = all_inputs.input_ids + [tokenizer.pad_token_id] * (encoder_lengths - len(all_inputs.input_ids))
                        inputs['encoder_input_ids'] = np.array(encoder_input_ids).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()

                        encoder_attention_mask = all_inputs.attention_mask + [0] * (encoder_lengths - len(all_inputs.input_ids))
                        inputs['encoder_attention_mask'] = np.array(encoder_attention_mask).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()

                        # inputs['input_ids'] = all_inputs['input_ids'][encoder_lengths:]
                        # inputs['attention_mask'] = all_inputs['attention_mask'][encoder_lengths:]

                        ## Add copy content as decoder input/target
                        start_id = random.randint(0, len(all_inputs["input_ids"]) - COPY_LEN)
                        copy_segments = all_inputs["input_ids"][start_id : start_id + COPY_LEN]

                        inputs["input_ids"] = copy_segments
                        inputs["attention_mask"] = [1 for _ in range(COPY_LEN)]
                        
                        labels = inputs["input_ids"].copy()
                        inputs["labels"] = labels
                        inputs["labels"][: -PRED_LEN] = [-100 for _ in range(len(labels) - PRED_LEN)]
                    
            elif input is not None:
                input = input.strip()

                # add llama2  template
                input = f'[INST]{input}[/INST]'
                output = output.strip() + tokenizer.eos_token

                # for too long inputs, we truncate first to save encoding time
                if len(input) > 5 * max_length:
                    input = input[:(5 * max_length) // 2] + input[-(5 * max_length) // 2:]

                tokenized_input = tokenizer.encode(input)
                tokenized_output = tokenizer.encode(output, add_special_tokens=False)
                output_length = len(tokenized_output)                
                # some outputs are too long, discard them
                if output_length > max_length:
                    return None

                # truncate from middle
                input_max_length = max_length - output_length

                if len(tokenized_input) > input_max_length:
                    half = int(input_max_length / 2)
                    input = tokenizer.decode(tokenized_input[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_input[-half:], skip_special_tokens=True)

                if with_labels:
                    pattern = rng.choice(patterns).tolist()
                    all_inputs = tokenizer(pattern.join([input, output]))
                    maybe_truncated_input=tokenizer.encode(input)
                    if len(all_inputs["input_ids"]) < min_length:
                        return None

                    #### split into encoder and decoder part
                    num_encoder_chunks = max(len(maybe_truncated_input) // CHUNK_SIZE - 1, 1)
                    encoder_lengths = num_encoder_chunks * CHUNK_SIZE

                    ### determine encoder input
                    inputs['encoder_input_ids'] = np.array(all_inputs.input_ids[:encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()
                    inputs['encoder_attention_mask'] = np.array(all_inputs.attention_mask[:encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()

                    inputs['input_ids'] = all_inputs['input_ids'][encoder_lengths:]
                    inputs['attention_mask'] = all_inputs['attention_mask'][encoder_lengths:]
                    #### Add-on ends

                    ### determine decoder input
                    labels = inputs["input_ids"].copy()
                    labels[:-output_length] = [-100 for _ in range(len(labels) - output_length)]
                    inputs["labels"] = labels
                else:
                    inputs = tokenizer(input)
                    raise ValueError('with labels must be true')

            elif input_ids is not None:
                # if len(input_ids) < min_length or len(input_ids) > max_length:
                #     return None
                # inputs = {
                #     "input_ids": input_ids,
                #     "labels": labels,
                # }
                raise NotImplementedError(f"Direct processing input_ids is not implemented")

            else:
                raise NotImplementedError(f"The dataset must contain one of the following fields: 'input' & 'output' (for fine-tuning), 'text' (for pre-training), 'input_ids' & 'labels' for passkey_retrieval.")

            # index is required for evaluation, by default we always add it
            inputs["index"] = _index
            # length is required for grouping
            inputs["length"] = len(inputs["input_ids"])
            return inputs
        return process_fn

    @staticmethod
    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, context_size=512, max_train_num_per_data=None, seed=42, cache_dir=None, thr=0.0):
        if data_files is None:
            return None
        if isinstance(data_files, str):
            if "*" in data_files:
                data_files = glob(data_files)
            else:
                data_files = [data_files]
        
        print(f'Processing dataset with {thr=}')
        process_fn = DataMixed.get_process_fn(tokenizer, max_length=max_length, min_length=min_length, context_size=context_size, seed=seed, thr=thr)

        train_datasets = []
        for path in data_files:
            temp_dataset = datasets.load_dataset('json', data_files=path, split='train', cache_dir=cache_dir)
            temp_dataset = temp_dataset.map(process_fn, batched=True, num_proc=96, batch_size=1280, remove_columns=temp_dataset.column_names)
            if max_train_num_per_data is not None and len(temp_dataset) > max_train_num_per_data:
                temp_dataset = temp_dataset.train_test_split(max_train_num_per_data, seed=seed)["test"]
            train_datasets.append(temp_dataset)

        dataset = datasets.concatenate_datasets(train_datasets)
        return dataset


class Datav2:
    # Dynamic Prefix/Suffix
    @staticmethod
    def get_process_fn(tokenizer, min_length, max_length, context_size, seed=42, with_labels=True):
        patterns = [" ", "\n", "\n\n"]
        rng = np.random.default_rng(seed=seed)
        CHUNK_SIZE = context_size

        @DatasetProcessFn()
        def process_fn(text=None, input=None, output=None, input_ids=None, labels=None, _index=None, **kwds):

            inputs = dict()
            if text is not None:
                PREFIX_SIZE, SUFFIX_SIZE = 1, 512
                # truncate text for faster processing
                text = text[:max_length * 5]
                if with_labels:
                    all_inputs = tokenizer(text, truncation=True, max_length=max_length)
                    if len(all_inputs["input_ids"]) < min_length:
                        return None

                    MAX_CHUNK = 8
                    MAX_NUM_CHUNK = min((8192 - 2048) // CHUNK_SIZE, MAX_CHUNK)

                    num_encoder_chunks = min(max(len(all_inputs.input_ids) // CHUNK_SIZE - 1, 1), MAX_NUM_CHUNK)
                    encoder_lengths = num_encoder_chunks * CHUNK_SIZE

                    inputs['encoder_input_ids'] = np.array(all_inputs.input_ids[PREFIX_SIZE:PREFIX_SIZE+encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()
                    inputs['encoder_attention_mask'] = np.array(all_inputs.attention_mask[PREFIX_SIZE:PREFIX_SIZE+encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()

                    inputs['input_ids'] = all_inputs['input_ids'][PREFIX_SIZE+encoder_lengths:]
                    inputs['attention_mask'] = all_inputs['attention_mask'][PREFIX_SIZE+encoder_lengths:]

                    # For context modeling, we do not use customized prefix/suffix
                    ### add prefix/suffix if required
                    inputs["encoder_prefix_id"] = all_inputs.input_ids[:PREFIX_SIZE]
                    inputs["encoder_suffix_id"] = all_inputs.input_ids[-SUFFIX_SIZE:]

                    #### Add-on ends
                    labels = inputs["input_ids"].copy()
                    ## MASK PREVIOUS TOKENS (add bound detection)
                    inputs["labels"] = labels
                    
            elif input is not None:
                PREFIX_SIZE, SUFFIX_SIZE = 1, 32
                input = input.strip()
                output = output.strip() + tokenizer.eos_token
                # for too long inputs, we truncate first to save encoding time
                if len(input) > 5 * max_length:
                    input = input[:(5 * max_length) // 2] + input[-(5 * max_length) // 2:]

                tokenized_input = tokenizer.encode(input)
                tokenized_output = tokenizer.encode(output, add_special_tokens=False)

                output_length = len(tokenized_output)                
                # some outputs are too long, discard them
                if output_length > max_length:
                    return None

                # truncate from middle
                input_max_length = max_length - output_length
                if len(tokenized_input) > input_max_length:
                    half = int(input_max_length / 2)
                    input = tokenizer.decode(tokenized_input[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_input[-half:], skip_special_tokens=True)

                if with_labels:
                    pattern = rng.choice(patterns).tolist()
                    all_inputs = tokenizer(pattern.join([input, output]))
                    maybe_truncated_input=tokenizer.encode(input)
                    if len(all_inputs["input_ids"]) < min_length:
                        return None

                    #### split into encoder and decoder part
                    num_encoder_chunks = max(len(maybe_truncated_input) // CHUNK_SIZE - 1, 1)
                    encoder_lengths = num_encoder_chunks * CHUNK_SIZE


                    inputs['encoder_input_ids'] = np.array(all_inputs.input_ids[PREFIX_SIZE:PREFIX_SIZE+encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()
                    inputs['encoder_attention_mask'] = np.array(all_inputs.attention_mask[PREFIX_SIZE:PREFIX_SIZE+encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()

                    inputs['input_ids'] = all_inputs['input_ids'][PREFIX_SIZE+encoder_lengths:]
                    inputs['attention_mask'] = all_inputs['attention_mask'][PREFIX_SIZE+encoder_lengths:]
                    #### Add-on ends

                    labels = inputs["input_ids"].copy()
                    labels[:-output_length] = [-100 for _ in range(len(labels) - output_length)]
                    inputs["labels"] = labels

                    ### add prefix/suffix if required
                    inputs["encoder_prefix_id"] = all_inputs.input_ids[:PREFIX_SIZE]
                    inputs["encoder_suffix_id"] = all_inputs.input_ids[-SUFFIX_SIZE-output_length:-output_length]
                    ### add-on ends
                else:
                    inputs = tokenizer(input)
                    raise ValueError('with labels must be true')

            elif input_ids is not None:
                raise NotImplementedError(f"Direct processing input_ids is not implemented")

            else:
                raise NotImplementedError(f"The dataset must contain one of the following fields: 'input' & 'output' (for fine-tuning), 'text' (for pre-training), 'input_ids' & 'labels' for passkey_retrieval.")

            # index is required for evaluation, by default we always add it
            inputs["index"] = _index
            # length is required for grouping
            inputs["length"] = len(inputs["input_ids"])
            return inputs
        return process_fn

    @staticmethod
    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, context_size=512, max_train_num_per_data=None, seed=42, cache_dir=None):
        if data_files is None:
            return None
        if isinstance(data_files, str):
            if "*" in data_files:
                data_files = glob(data_files)
            else:
                data_files = [data_files]
        
        process_fn = Datav2.get_process_fn(tokenizer, max_length=max_length, min_length=min_length, context_size=context_size, seed=seed)

        train_datasets = []
        for path in data_files:
            temp_dataset = datasets.load_dataset('json', data_files=path, split='train', cache_dir=cache_dir)
            temp_dataset = temp_dataset.map(process_fn, batched=True, num_proc=96, batch_size=1280, remove_columns=temp_dataset.column_names)

            if max_train_num_per_data is not None and len(temp_dataset) > max_train_num_per_data:
                temp_dataset = temp_dataset.train_test_split(max_train_num_per_data, seed=seed)["test"]
            train_datasets.append(temp_dataset)

        dataset = datasets.concatenate_datasets(train_datasets)
        return dataset

class Datav2Mixed:
    # Add completion task
    @staticmethod
    def get_process_fn(tokenizer, min_length, max_length, context_size, seed=42, with_labels=True, thr=0.0):
        patterns = [" ", "\n", "\n\n"]
        rng = np.random.default_rng(seed=seed)
        CHUNK_SIZE = context_size

        @DatasetProcessFn()
        def process_fn(text=None, input=None, output=None, input_ids=None, labels=None, _index=None, **kwds):

            inputs = dict()
            if text is not None:
                PREFIX_SIZE, SUFFIX_SIZE = 1, 32
                REAL_CHUNK_SIZE = CHUNK_SIZE - PREFIX_SIZE - SUFFIX_SIZE
                # truncate text for faster processing
                text = text[:max_length * 5]
                if with_labels:
                    all_inputs = tokenizer(text, truncation=True, max_length=max_length)
                    if len(all_inputs["input_ids"]) < min_length:
                        return None

                    MAX_CHUNK = 8
                    MAX_NUM_CHUNK = min((8192 - 1024) // CHUNK_SIZE, MAX_CHUNK)
                    import random
                    if random.random() > thr:
                        ####################
                        ## continualation ##
                        ####################
                        num_encoder_chunks = min(max(len(all_inputs.input_ids) // CHUNK_SIZE - 1, 1), MAX_NUM_CHUNK)
                        encoder_lengths = num_encoder_chunks * CHUNK_SIZE

                        inputs['encoder_input_ids'] = np.array(all_inputs.input_ids[PREFIX_SIZE:PREFIX_SIZE+encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()
                        inputs['encoder_attention_mask'] = np.array(all_inputs.attention_mask[PREFIX_SIZE:PREFIX_SIZE+encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()

                        inputs['input_ids'] = all_inputs['input_ids'][PREFIX_SIZE+encoder_lengths:]
                        inputs['attention_mask'] = all_inputs['attention_mask'][PREFIX_SIZE+encoder_lengths:]

                        # For context modeling, we do not use customized prefix/suffix
                        ### add prefix/suffix if required
                        inputs["encoder_prefix_id"] = all_inputs.input_ids[:PREFIX_SIZE]
                        inputs["encoder_suffix_id"] = all_inputs.input_ids[-SUFFIX_SIZE:]

                        #### Add-on ends
                        labels = inputs["input_ids"].copy()
                        ## MASK PREVIOUS TOKENS (add bound detection)
                        inputs["labels"] = labels
                    else:
                        ######################
                        ## repeatition loss ##
                        ######################
                        COPY_LENGTH = 1024

                        # REAL_CHUNK_SIZE = CHUNK_SIZE
                        PREFIX_SIZE, SUFFIX_SIZE = 1, 31
                        REAL_CHUNK_SIZE = CHUNK_SIZE - PREFIX_SIZE - SUFFIX_SIZE

                        # Step 1: random pick "input segments" as suffix
                        copy_start_pos = random.randint(PREFIX_SIZE, len(all_inputs["input_ids"]) - COPY_LENGTH)
                        suffix_id = all_inputs['input_ids'][copy_start_pos : copy_start_pos + SUFFIX_SIZE]
                        inputs['input_ids'] = all_inputs['input_ids'][copy_start_pos:copy_start_pos+COPY_LENGTH]
                        inputs['attention_mask'] = all_inputs['attention_mask'][copy_start_pos:copy_start_pos+COPY_LENGTH]

                        ### add prefix/suffix (default)
                        inputs["encoder_prefix_id"] = all_inputs.input_ids[:PREFIX_SIZE]
                        inputs["encoder_suffix_id"] = suffix_id

                        # construct input_ids for encoder and decoder
                        # put all context into encoder
                        from math import ceil
                        num_encoder_chunks = ceil(len(all_inputs.input_ids[PREFIX_SIZE:]) / REAL_CHUNK_SIZE)
                        encoder_lengths = num_encoder_chunks * REAL_CHUNK_SIZE

                        # inputs['encoder_input_ids'] = np.array(all_inputs.input_ids[PREFIX_SIZE:PREFIX_SIZE+encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()
                        # inputs['encoder_attention_mask'] = np.array(all_inputs.attention_mask[PREFIX_SIZE:PREFIX_SIZE+encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()
                        pad_length = encoder_lengths - (len(all_inputs.input_ids) - PREFIX_SIZE)
                        inputs['encoder_input_ids'] = np.array(all_inputs.input_ids[PREFIX_SIZE:] + [tokenizer.pad_token_id] * pad_length).reshape(num_encoder_chunks, REAL_CHUNK_SIZE).tolist()
                        inputs['encoder_attention_mask'] = np.array(all_inputs.attention_mask[PREFIX_SIZE:] + [0] * pad_length).reshape(num_encoder_chunks, REAL_CHUNK_SIZE).tolist()

                        # Step 3: construct labels
                        # append suffix ids to input_ids (as copy loss)
                        inputs["input_ids"] = inputs['input_ids'] + suffix_id
                        inputs["attention_mask"] = inputs['attention_mask'] + [1] * SUFFIX_SIZE

                        labels = inputs["input_ids"].copy()
                        ## mask suffix tokens (serve as queries)
                        labels[:SUFFIX_SIZE] = [-100 for _ in range(SUFFIX_SIZE)]
                        inputs["labels"] = labels
                        
                    
            elif input is not None:
                PREFIX_SIZE, SUFFIX_SIZE = 1, 32
                input = input.strip()
                output = output.strip() + tokenizer.eos_token
                # for too long inputs, we truncate first to save encoding time
                if len(input) > 5 * max_length:
                    input = input[:(5 * max_length) // 2] + input[-(5 * max_length) // 2:]

                tokenized_input = tokenizer.encode(input)
                tokenized_output = tokenizer.encode(output, add_special_tokens=False)

                output_length = len(tokenized_output)                
                # some outputs are too long, discard them
                if output_length > max_length:
                    return None

                # truncate from middle
                input_max_length = max_length - output_length
                if len(tokenized_input) > input_max_length:
                    half = int(input_max_length / 2)
                    input = tokenizer.decode(tokenized_input[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_input[-half:], skip_special_tokens=True)

                if with_labels:
                    pattern = rng.choice(patterns).tolist()
                    all_inputs = tokenizer(pattern.join([input, output]))
                    maybe_truncated_input=tokenizer.encode(input)
                    if len(all_inputs["input_ids"]) < min_length:
                        return None

                    #### split into encoder and decoder part
                    num_encoder_chunks = max(len(maybe_truncated_input) // CHUNK_SIZE - 1, 1)
                    encoder_lengths = num_encoder_chunks * CHUNK_SIZE


                    inputs['encoder_input_ids'] = np.array(all_inputs.input_ids[PREFIX_SIZE:PREFIX_SIZE+encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()
                    inputs['encoder_attention_mask'] = np.array(all_inputs.attention_mask[PREFIX_SIZE:PREFIX_SIZE+encoder_lengths]).reshape(num_encoder_chunks, CHUNK_SIZE).tolist()

                    inputs['input_ids'] = all_inputs['input_ids'][PREFIX_SIZE+encoder_lengths:]
                    inputs['attention_mask'] = all_inputs['attention_mask'][PREFIX_SIZE+encoder_lengths:]
                    #### Add-on ends

                    labels = inputs["input_ids"].copy()
                    labels[:-output_length] = [-100 for _ in range(len(labels) - output_length)]
                    inputs["labels"] = labels

                    ### add prefix/suffix if required
                    inputs["encoder_prefix_id"] = all_inputs.input_ids[:PREFIX_SIZE]
                    inputs["encoder_suffix_id"] = all_inputs.input_ids[-SUFFIX_SIZE-output_length:-output_length]
                    ### add-on ends
                else:
                    inputs = tokenizer(input)
                    raise ValueError('with labels must be true')

            elif input_ids is not None:
                raise NotImplementedError(f"Direct processing input_ids is not implemented")

            else:
                raise NotImplementedError(f"The dataset must contain one of the following fields: 'input' & 'output' (for fine-tuning), 'text' (for pre-training), 'input_ids' & 'labels' for passkey_retrieval.")

            # index is required for evaluation, by default we always add it
            inputs["index"] = _index
            # length is required for grouping
            inputs["length"] = len(inputs["input_ids"])
            return inputs
        return process_fn

    @staticmethod
    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, context_size=512, max_train_num_per_data=None, seed=42, cache_dir=None, thr=0.0):
        if data_files is None:
            return None
        if isinstance(data_files, str):
            if "*" in data_files:
                data_files = glob(data_files)
            else:
                data_files = [data_files]
        
        print(f'Processing dataset with {thr=}\n')
        process_fn = Datav2Mixed.get_process_fn(tokenizer, max_length=max_length, min_length=min_length, context_size=context_size, seed=seed, thr=thr)

        train_datasets = []
        for path in data_files:
            temp_dataset = datasets.load_dataset('json', data_files=path, split='train', cache_dir=cache_dir)
            temp_dataset = temp_dataset.map(process_fn, batched=True, num_proc=96, batch_size=1280, remove_columns=temp_dataset.column_names)

            if max_train_num_per_data is not None and len(temp_dataset) > max_train_num_per_data:
                temp_dataset = temp_dataset.train_test_split(max_train_num_per_data, seed=seed)["test"]
            train_datasets.append(temp_dataset)

        dataset = datasets.concatenate_datasets(train_datasets)
        return dataset

def get_max_length_in_nested_lists(lst):
    if len(lst) and isinstance(lst[0], list):
        lengths = []
        for elem in lst:
            length = get_max_length_in_nested_lists(elem)
            lengths.append(length)
        max_length = max(lengths)
        return max_length
    else:
        return len(lst)

def pad_nested_lists(lst, max_length, padding_value, padding_side="right"):
    if isinstance(lst, list) and len(lst) and isinstance(lst[0], list):
        masks = []
        for i, elem in enumerate(lst):
            lst[i], mask = pad_nested_lists(elem, max_length, padding_value, padding_side)
            masks.append(mask)
        return lst, masks
    elif isinstance(lst, list):
        if padding_side == "right":
            mask = [1] * len(lst) + [0] * (max_length - len(lst))
            lst = lst + [padding_value for _ in range(max_length - len(lst))]
            return lst, mask
        else:
            mask = [0] * (max_length - len(lst)) + [1] * len(lst)
            lst = [padding_value for _ in range(max_length - len(lst))] + lst
            return lst, mask
    else:
        raise NotImplementedError(f"Unrecognized type {lst}")

@dataclass
class DefaultDataCollator:
    """
    Data collator that can:
    1. Dynamically pad all inputs received. The inputs must be dict of lists.
    2. Add position_ids based on attention_mask if required.
    """
    tokenizer: PreTrainedTokenizer
    attention_padding_value: int = 0
    label_padding_value: int = -100
    add_position_ids: bool = False

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        first_elem = batch_elem[0]
        bsz = len(batch_elem)
        return_batch = {}
        
        for key, value in first_elem.items():
            # print(key)
            # HACK: any key containing attention_mask must be attention_mask
            # important to assign different pad token for different types of inputs
            if "attention_mask" in key:
                pad_token_id = self.attention_padding_value
            elif "label" in key:
                pad_token_id = self.label_padding_value
            else:
                pad_token_id = self.tokenizer.pad_token_id

            batch_value = [elem[key] for elem in batch_elem]
            # pad all lists and nested lists
            if key not in ['prefix_id', 'suffix_id']:
                if isinstance(value, list):
                    max_length = get_max_length_in_nested_lists(batch_value)
                    batch_value, _ = pad_nested_lists(batch_value, max_length, pad_token_id, self.tokenizer.padding_side)

            try:
                return_batch[key] = torch.tensor(batch_value)
            except:
                # handle strings and None
                return_batch[key] = batch_value

            if "attention_mask" in key and self.add_position_ids:
                value = return_batch[key]
                position_ids = value.cumsum(-1) - 1
                position_ids = position_ids.masked_fill(value == 0, 0)
                return_batch[key.replace("attention_mask", "position_ids")] = position_ids

        return return_batch