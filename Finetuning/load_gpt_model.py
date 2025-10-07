FROM_REMOTE = True
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils import *
#from finnlp.benchmarks.tfns import test_tfns
import logging
from accelerate import disk_offload
# Suppress Warnings during inference
logging.getLogger("transformers").setLevel(logging.ERROR)



def load_model(base_model, peft_model, from_remote=False):
    
    model_name = parse_model_name(base_model, from_remote)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, 
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    tokenizer.padding_side = "left"
    if base_model == 'qwen':
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(model, peft_model)
    model = model.eval()
    #model = disk_offload(model)
    return model, tokenizer