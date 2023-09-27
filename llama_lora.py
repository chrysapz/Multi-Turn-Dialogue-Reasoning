# https://huggingface.co/docs/peft/quicktour
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from transformers import AutoModelForSeq2SeqLM

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05,
    bias = 'none', target_modules = ["q", "v"]
) #! maybe we should add target_modules but I am not sure that the allowed values are the same for every model

model = AutoModelForSeq2SeqLM.from_pretrained('bigscience/mt0-small')
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

#Now you can train the PeftModel as you normally would train the base model.