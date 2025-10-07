import os
import sys
import argparse
from datetime import datetime
from functools import partial
import datasets
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import gc


from transformers.trainer import TRAINING_ARGS_NAME
from transformers.integrations import TensorBoardCallback
# Importing LoRA specific modules
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)
from utils import *

from accelerate.utils import DistributedType


acces_token = ""
os.environ['WANDB_API_KEY'] = ''
os.environ['WANDB_PROJECT'] = ''


def main(args):
    """
    Main function to execute the training script.

    :param args: Command line arguments
    """

    # Parse the model name and determine if it should be fetched from a remote source
    model_name = parse_model_name(args.base_model, args.from_remote)
    print(f"model_name : {model_name}")
    # Load the pre-trained causal language model
    if args.base_model in ["llama3","llama31","llama31Inst8B","llama3Inst70B","llama31Inst70B"]:
        model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #load_in_8bit=True,
        #device_map="auto",
        #device_map={'': 'cuda:0'},
        trust_remote_code=True,
        token = acces_token
        )
    elif args.base_model == 'mt0-xl':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True
            #device_map= 'auto'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)
        
    # Print model architecture for the first process in distributed training
    if args.local_rank == 0:
        print(model)

    # Load tokenizer associated with the pre-trained model
    if args.base_model in ["llama3","llama31","llama3Inst70B","llama31Inst70B","llama31Inst8B"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token = acces_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Apply model specific tokenization settings
    if args.base_model != 'mpt':
        tokenizer.padding_side = "left"
    if args.base_model == 'qwen':
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')
    # Ensure padding token is set correctly
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    # Load training and testing datasets
    dataset_list = load_dataset(args.dataset, args.from_remote)
    dataset_train = datasets.concatenate_datasets([d['train'] for d in dataset_list]).shuffle(seed=42)
    
    if args.test_dataset:
        dataset_list = load_dataset(args.test_dataset, args.from_remote)
    dataset_test = datasets.concatenate_datasets([d['test'] for d in dataset_list])
    
    dataset = datasets.DatasetDict({'train': dataset_train, 'test': dataset_test})
    # Display first sample from the training dataset
    print(dataset['train'][0])
    # Filter out samples that exceed the maximum token length and remove unused columns
    dataset = dataset.map(partial(tokenize, args, tokenizer))
    print('original dataset length: ', len(dataset['train']))
    dataset = dataset.filter(lambda x: not x['exceed_max_length'])
    print('filtered dataset length: ', len(dataset['train']))
    dataset = dataset.remove_columns(['instruction', 'input', 'output', 'exceed_max_length'])
    #dataset = dataset.remove_columns(['instruction', 'input', 'output'])
    print(dataset['train'][0])

    # Create a timestamp for model saving
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f'finetuned_models/{args.run_name}_{formatted_time}', # 保存位置
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_steps,
        dataloader_num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        fp16=True,
        # fp16_full_eval=True,
        deepspeed=args.ds_config,
        eval_strategy=args.evaluation_strategy,
        load_best_model_at_end=args.load_best_model,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=args.run_name
    )
    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    if not args.base_model == 'mpt':
        #model.gradient_checkpointing_enable(gradient_checkpointing_func=my_gradient_checkpointing_function)
        #model.gradient_checkpointing_enable()
        pass
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = (
        False
    )
    # model = prepare_model_for_int8_training(model
#SEQ_CLS = "SEQ_CLS"  --> Text classification
#SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"  --> Sequence-to-sequence language modeling.
#CAUSAL_LM = "CAUSAL_LM"  --> Causal language modeling.
#TOKEN_CLS = "TOKEN_CLS"  --> Token classification.
#QUESTION_ANS = "QUESTION_ANS"  --> Question answering.
#FEATURE_EXTRACTION = "FEATURE_EXTRACTION"  --> Feature extraction. Provides the hidden states which can be used as embeddings or features for downstream tasks. 
# lora_dropout = [0.01, ..., 0.1]
 
    # setup peft for lora
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=lora_module_dict[args.base_model],
        #target_modules = "all-linear",
        bias='none',
    )
    model = get_peft_model(model, peft_config)
    model.generation_config.max_new_tokens = 6
    model.generation_config.min_new_tokens = 1
    # Initialize TensorBoard for logging
    writer = SummaryWriter()

    # Initialize the trainer
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"], 
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            #padding='max_length',
            return_tensors="pt"
        ),
        callbacks=[TensorBoardCallback(writer)],
    )
    
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # Clear CUDA cache and start training
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()
    writer.close()

    # Save the fine-tuned model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=['fingpt_glm','chatglm2', 'llama2', 'llama2-13b', 'llama2-13b-nr','llama3','llama31','llama31Inst8B','llama3Inst70B','llama31Inst70B', 'DeepSeek-R1-Distill-Llama-8B','DeepSeek-R1', 'mt0-xl',  'baichuan', 'falcon', 'falcon-1b', 'internlm', 'qwen', 'mpt', 'bloom','bloomberg','DeciLM-7B', "flan-t5-base"])
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="The train batch size per device")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate")
    parser.add_argument("--num_epochs", default=8, type=float, help="The training epochs")
    parser.add_argument("--gradient_steps", default=8, type=float, help="The gradient accumulation steps")
    parser.add_argument("--num_workers", default=8, type=int, help="dataloader workers")
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--ds_config", default='./config_new.json', type=str)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--evaluation_strategy", default='steps', type=str)
    parser.add_argument("--load_best_model", default='False', type=bool)
    parser.add_argument("--eval_steps", default=0.1, type=float)    
    parser.add_argument("--from_remote", default=False, type=bool)   
    #parser.add_argument("--num_processes", default=1, type=int)
    args = parser.parse_args()

    # Login to Weights and Biases
    wandb.login()

    # Run the main function
    main(args)
