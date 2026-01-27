from unsloth import FastLanguageModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_sharegpt,
    train_on_responses_only,
)

# isort: split
from functools import partial

import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, TextStreamer
from trl.trainer.sft_trainer import  SFTTrainer
from trl.trainer.sft_config import SFTConfig

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
    # "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    # "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    # "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # 4bit for 405b!
    # "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
    # "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    # "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    # "unsloth/Phi-3-medium-4k-instruct",
    # "unsloth/gemma-2-9b-bnb-4bit",
    # "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
    # "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
    # "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    # "unsloth/Llama-3.2-3B-bnb-4bit",
    # "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    # "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",  # NEW! Llama 3.3 70B!
]

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

def get_model():

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",  # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    return model, tokenizer


def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {
        "text": texts,
    }


def get_trainer():
    dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    # convert sharegpt to hugging face format

    dataset = standardize_sharegpt(dataset)
    model, tokenizer = get_model()
    dataset = dataset.map(
        partial(formatting_prompts_func, tokenizer=tokenizer),
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,  # Can make training 5x faster for short sequences.
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=60,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use TrackIO/WandB etc
            bf16=False,  # my pc does not support it
        ),
    )
    # train_on_completion:
    # only train on the assistant outputand ingore the loss on the user input
    # you don't want to learn how to predict the user input
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    tokenizer.decode(trainer.train_dataset[5]["input_ids"])

    # verify that masking is done
    space = tokenizer(" ", add_special_tokens=False).input_ids[0]
    tokenizer.decode(
        [space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]
    )
    return trainer


def show_current_memory_stat(trainer):
    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


"""<a name="Inference"></a>
### Inference
Let's run the model! You can change the instruction and input - leave the output blank!



We use  for more information on why.
"""


def train(model, tokenizer):
    # `min_p = 0.1` and `temperature = 1.5`. Read this [Tweet](https://x.com/menhguin/status/1826132708508213629)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    messages = [
        {
            "role": "user",
            "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs, max_new_tokens=64, use_cache=True, temperature=1.5, min_p=0.1
    )
    tokenizer.batch_decode(outputs)
    # use text streamer to see token by token generation

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    messages = [
        {
            "role": "user",
            "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )


def save_lora_adapters(model, tokenizer):
    # only saves adapters not the models

    model.save_pretrained("lora_model")  # Local saving
    tokenizer.save_pretrained("lora_model")


"""Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:"""


def reload_models(model_name="lora_model"):
    if False:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

        messages = [
            {
                "role": "user",
                "content": "Describe a tall tower in the capital of France.",
            },
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )


def realod_models_huggingface():
    if False:
        # I highly do NOT suggest - use Unsloth if possible, is slow
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        model = AutoPeftModelForCausalLM.from_pretrained(
            "lora_model",  # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit=load_in_4bit,
        )
        tokenizer = AutoTokenizer.from_pretrained("lora_model")


def save_models(model, tokenizer):
    # Merge to 16bit
    if False:
        model.save_pretrained_merged(
            "model",
            tokenizer,
            save_method="merged_16bit",  # merged_4bit for int4
        )
    if False:
        model.push_to_hub_merged(
            "hf/model", tokenizer, save_method="merged_16bit", token=""
        )

    # Merge to 4bit
    if False:
        model.save_pretrained_merged(
            "model",
            tokenizer,
            save_method="merged_4bit",
        )
    if False:
        model.push_to_hub_merged(
            "hf/model", tokenizer, save_method="merged_4bit", token=""
        )

    # Just LoRA adapters
    if False:
        model.save_pretrained("model")
        tokenizer.save_pretrained("model")


def save_model_gguf(model, tokenizer):
    # Save to 8bit Q8_0
    if False:
        # gguf -> binary file format to save local llms
        # canhave different quantization methods
        #     * `q8_0` - Fast conversion. High resource use, but generally acceptable.
        # * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
        # * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
        model.save_pretrained_gguf(
            "model",
            tokenizer,
        )

    # Save to 16bit GGUF
    if False:
        model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
    if False:
        model.push_to_hub_gguf(
            "hf/model", tokenizer, quantization_method="f16", token=""
        )

    # Save to q4_k_m GGUF
    if False:
        model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
    if False:
        model.push_to_hub_gguf(
            "hf/model", tokenizer, quantization_method="q4_k_m", token=""
        )

    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "hf/model",  # Change hf to your username!
            tokenizer,
            quantization_method=[
                "q4_k_m",
                "q8_0",
                "q5_k_m",
            ],
            token="",  # Get a token at https://huggingface.co/settings/tokens
        )


trainer = get_trainer()
trainer_stats = trainer.train()
