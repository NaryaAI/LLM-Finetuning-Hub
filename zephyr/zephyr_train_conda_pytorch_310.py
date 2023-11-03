#!/usr/bin/env python
# coding: utf-8

# # TRAIN
# This script is an example of how you can fine-tune **HuggingFaceH4/zephyr-7b-alpha** on **Narya-ai/title_subtitle** dataset \
# * Use peft and quantization to fit the model for training on the smaller GPU instance \
# * Save the checkpoints and the pefted model \
# * Push to Huggingface hub \
#
#
# For Inference see **zephyr_inference_conda_pytorch_310.ipynb**

# In[2]:


# !pip install torch transformers[torch] datasets
# !pip install accelerate -U
# !pip install peft
# !pip install trl
# !pip install bitsandbytes
# !pip install ninja packaging
# !MAX_JOBS=32 pip install flash-attn --no-build-isolation


# In[1]:


# !huggingface-cli login --token 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
# !huggingface-cli login --token <TOKEN>


# In[2]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline, AutoModel, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import gc
torch.cuda.empty_cache()
gc.collect()


# ### Load original Zephyr model and Zephyr tokenizer

# In[3]:


model_name = "HuggingFaceH4/zephyr-7b-alpha"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map="auto",

    # IMPORTANT, FA2 gives 3+x speed up
    # https://github.com/Dao-AILab/flash-attention
    # https://github.com/huggingface/peft/issues/790#issuecomment-1696801352
    use_flash_attention_2=True,
)


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
# model.config.pretraining_tp = 1  # uncomment if you call LlaMa-2

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# tokenizer.enable_padding()

# IMPORTANT IF ON SAGEMAKER NOTEBOOK. set path to folder that is on a LARGE disk
# otherwise disk get's full fast and then gives "No space left error"
# can check the disks spaces with "df -h" on terminal
cache_dir = "/home/ec2-user/SageMaker/cache"

# if all good the model weights must be distributed across all available gpus
model.hf_device_map


# ### Load **Narya-ai/title_subtitle** dataset

# In[4]:


def convert_data(d):
    messages = [
        {"role": "system", "content": d['system_message']},
        {"role": "user", "content": d['user_message']},
        {"role": "assistant", "content": d['completion']},
    ]
    return {'text': tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)}

def tokenize_function(d):
    source_encodings = tokenizer(d['text'], return_tensors="pt",
                                 padding='max_length', truncation=True, max_length=2500)
    source_encodings["labels"] = source_encodings["input_ids"]
    return source_encodings

dataset_path = 'title_subtitle_dataset.jsonl'
dataset = load_dataset('json', data_files={'train': dataset_path}, split='train')
dataset = dataset.map(convert_data, remove_columns=['system_message', 'user_message', 'completion'])
dataset = dataset.map(
    tokenize_function,
    batched=True,
)

train_dataset, eval_dataset = dataset.train_test_split(test_size=0.2).values()
dataset


# In[5]:


training_args = TrainingArguments(
    # IMPORTANT Adjust based on your GPU's capabilities
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    # IMPORTANT local folder to save checkpoints. if call trainer.save_model saves to this folder
    # trainer.push_to_hub uploads from this folder
    output_dir="./some_output",
    # IMPORTANT if you use trainer.push_to_hub it uploads to this repo-id
    hub_model_id="Narya-ai/zephyr-title-subtitle",
    save_steps=50,
    load_best_model_at_end=True,
    num_train_epochs=10,
    logging_dir='./logs',
    logging_steps=50,
    eval_steps=50,
    evaluation_strategy="steps",
    save_total_limit=1,
    push_to_hub=False,  # Set to True if you want to push to HuggingFace's Model Hub
    fp16=True,  # Use mixed precision for faster training (ensure you have the `apex` library installed)
    gradient_accumulation_steps=10,
    optim="paged_adamw_8bit",
    hub_private_repo=True,
)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=2500,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)
model.config.use_cache = False


# In[6]:


trainer.train(resume_from_checkpoint=True)


# ## There are 2 ways to save and upload to hub the fine-tuned model
# ### 1. Via Trainer(recommended)
#
#
# Note:
# * uploads only necessary files including adapters, tokenizer, etc.(NOT THE MODEL ITSELF)
# * It uploads the files saved in output_dir that you set in TrainingArguments
# * It uses the hub_model_id (user_name/model_name) that you set in TrainingArguments. The default is [here](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/trainer#transformers.TrainingArguments.hub_model_id)

# In[ ]:


trainer.push_to_hub()


# # If you wanted to only fine-tune and upload/save model STOP HERE

# # This is just to understand what is happening to the model and what is being uploaded by different functions

# In[ ]:


# s_weight = sum(p.numel() for p in model.parameters())
# print(f"model: {s_weight} after training")

# for name in ["HuggingFaceH4/zephyr-7b-alpha", "Narya-ai/zephyr-title-subtitle", "Narya-ai/zephyr-title-subtitle-full"]:
#     model = AutoModelForCausalLM.from_pretrained(
#         name,
#         load_in_8bit=True,
#         device_map="auto",
#         cache_dir=cache_dir,
#     )
#     s_weight = sum(p.numel() for p in model.parameters())
#     print(f"{name}: {s_weight}")

# model = AutoModelForCausalLM.from_pretrained(
#     "HuggingFaceH4/zephyr-7b-alpha",
#     load_in_8bit=True,
#     device_map="auto",
#     cache_dir=cache_dir,
# )
# adapter_name = model.load_adapter("Narya-ai/zephyr-title-subtitle")
# model.active_adapters = adapter_name
# s_weight = sum(p.numel() for p in model.parameters())
# print(f"merging with original: {s_weight}")


# s_weight = sum(p.numel() for p in trainer.model.parameters())
# print(f"trainer.model: {s_weight}")


# In[ ]:


# This is to give a sense of what is happening with the weights
# Note the numbers might change after each training, but the pattern must stay the same
# summing weights using sum(p.numel() for p in model.parameters()) on models:

"""
ORIGINAL MODEL - NOT FINE-TUNED
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-alpha",
    load_in_8bit=True,
    device_map="auto",
    cache_dir=cache_dir,
)
7241732096
"""

"""
ONLY ADAPTERS
model = AutoModelForCausalLM.from_pretrained(
    "Narya-ai/zephyr-title-subtitle",
    load_in_8bit=True,
    device_map="auto",
    cache_dir=cache_dir,
)
7284252672
"""

"""
ONLY THE MODEL
model = AutoModelForCausalLM.from_pretrained(
    "Narya-ai/zephyr-title-subtitle-full",
    load_in_8bit=True,
    device_map="auto",
    cache_dir=cache_dir,
)
7241732096
"""


"""
Merging adapters to the original function
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-alpha",
    load_in_8bit=True,
    device_map="auto",
    cache_dir=cache_dir,
)
adapter_name = model.load_adapter("Narya-ai/zephyr-title-subtitle")
model.active_adapters = adapter_name
7284252672
"""


# ### 2. Via model.push_to_hub
# Note:
# * To upload the fine-tuned model(not original) you need to merge adapters first
# * This function will only push the ORIGINAL model, you'll need to push tokenizer too
# * This will upload the whole model, so it will be heavier memory wise


# cache_dir = "/home/ec2-user/SageMaker/cache"  # make sure there is enough space
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="cpu",  why cpu? see https://github.com/artidoro/qlora/issues/29#issuecomment-1737072311
#     cache_dir=cache_dir,
# )
#
# adapter = PeftModel.from_pretrained(
#     model=model,
#     model_id="./some_output/checkpoint-200",
#     torch_dtype=torch.float16,
#     device_map="cpu",  # why cpu? see https://github.com/artidoro/qlora/issues/29#issuecomment-1737072311
#
# )
# model = adapter.merge_and_unload(progressbar=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.save_pretrained(
#     "Narya-ai/zephyr-title-subtitle",
#     push_to_hub=True,
#     repo_id="Narya-ai/zephyr-title-subtitle",
#     private=True,
#     max_shard_size="4GB"
#
# )
# tokenizer.save_pretrained(
#     "Narya-ai/zephyr-title-subtitle",
#     push_to_hub=True,
#     repo_id="Narya-ai/zephyr-title-subtitle",
#     private=True,
# )

# ### Quick inference check

# In[ ]:


# pipe = pipeline("text-generation", model="Narya-ai/zephyr-title-subtitle",
#                 torch_dtype=torch.bfloat16, device_map="auto", tokenizer=tokenizer)

# prompt = '<|system|>\n\nIn this task you need to identify TITLE and SUBTITLE of a webpage that contains a web article.\nGuidelines:\n- TITLE is always present while SUBTITLE can be missing often\n- TITLE describes the main story of the article, while SUBTITLE usually is akin to an explanation of clarification of what the article is about, sometimes giving motivation to the story\n- SUBTITLE cannot be a list of author names, or publication date\n- SUBTITLE does not begin with words "abstract" or "summary"\n- SUBTITLE  cannot be larger than around 3 sentences\n- Research papers should only have TITLE, no SUBTITLE\n- If you see an obvious error in TITLE or SUBTITLE wording fix it\n\nYou have the following inputs:\n1) SUMMARY - a summary of the article so you understand what its about. Use it to check whether resulting TITLE and SUBTITLE make sense.\n2) SOURCE_URL - the source url for the webpage, sometimes using that you can understand roughly what the article main description is.\n3) SUGGESTED_TITLE - this is the prediction from an html-based extractor. A lot of the times its correct but sometimes its wrong\n4) SUGGESTED_SUBTITLE - this is the prediction from an html-based extractor. This is fairly often wrong\n5) FONTHTML - the list of html-derived text boxes with corresponding positions and fonts. It looks like a list of tuples, each tuple format is like this:\n(\'Some text\', \'14px\', \'400\', \'24px\', 40, 9696.625, 1120, 24)\nHere first item is text, then we have fontSize, fontWeight, lineHeight, x, y, width and height. Use this information to identify likely pieces of TITLE/SUBTITLE as they appear on top of the image, and sometimes are larger fonts\n\nThe correct guess for SUBTITLE should always follow the TITLE in the page. If in between there is any info - author information, date, image captions etc - then its not SUBTITLE.\n\nSUGGESTED_TITLE  is an ok option for best guess for TITLE when the OCR_text contains only rubbish, but please make sure it follows the guidelines above.\n\nPlease return TITLE and SUBTITLE only as a structured dictionary with two keys: "TITLE" and "SUBTITLE"\n</s>\n<|user|>\n\nSUMMARY:\nLena anderson, a fictional personality played by artificial intelligence (AI) software powering chatbots like OpenAIasOpenaiasGpt, discusses soccer with three real-world bot characters from fantasy. They discuss ways to reach new audiences for Major League Soccer (Mls), including augmented reality features, agamificationa, and the use of natural language models in an augmented reality mobile app. The bots, developed by fantasy\'s synthetic humans, can be used to test product concepts, generate new ideas, and help clients develop products. Another bot, trained on openAIas gpt - 4 language model, uses machine learning technology that powers chatbots such as openaias Gpt and GoogleasBard. It also builds chatbots based on data from diverse populations. A third bot simulates market behavior using openAIAS gpt4s\' neural networks. This is similar to the stanford university\'s genetic engineering research, which led to the creation of \'Synthetic Humans\', but they do not have access to existing product lines or knowledge of specific product lines. Despite these efforts, however, there are concerns about how to make this approach more effective. These discussions highlight the need for companies to invest in R&D to improve their AI capabilities, especially due to the growth of online gaming. Lena anderson suggests investing in projects like AR/VR, targeted advertising, ecommerce, and social media marketing, and blockchain funding, while other companies may offer additional funding.\nSOURCE_URL:\nhttps://www.wired.com/story/fast-forward-the-chatbots-are-now-talking-to-each-other/\nSUGGESTED_TITLE:\nThe Chatbots Are Now Talking to Each Other\nSUGGESTED_SUBTITLE:\nThe Chatbots Are Now Talking to Each Other\nFONT_HTML:\n[(\'Skip to main content\', \'19px\', \'400\', \'28px\', 0, 56, 1, 1), (\'Will Knight\', \'11px\', \'400\', \'12.98px\', 56, 112, 69.890625, 12), (\'Business\', \'11px\', \'400\', \'11px\', 151.890625, 112, 64.4375, 12), (\'Oct 12, 2023 11:00 AM\', \'11px\', \'400\', \'12.98px\', 232.328125, 104, 128.015625, 20.96875), (\'The Chatbots Are Now Talking to Each Other\', \'46.8px\', \'700\', \'46.8px\', 48, 166.96875, 1104, 62.796875), (\'ChatGPT-style chatbots that pretend to be people are being used to help companies develop new product and marketing ideas.\', \'20px\', \'700\', \'24px\', 48, 229.765625, 1104, 48), (\'Photograph: Carol Yepes/Getty Images\', \'11px\', \'400\', \'12.98px\', 48, 894.75, 264.375, 12), (\'Lena Anderson isn’t a soccer fan, but she does spend a lot of time ferrying her kids between soccer practices and competitive games.\', \'19px\', \'400\', \'28px\', 137.328125, 973.703125, 504, 84), (\'“I may not pull out a foam finger and painted face, but soccer does have a place in my life,” says the soccer mom—who also happens to be completely made up. Anderson is a fictional personality played by artificial intelligence software like that powering ChatGPT.\', \'19px\', \'400\', \'28px\', 137.328125, 1076.703125, 504, 140), (\'Content\', \'19px\', \'700\', \'28px\', 137.328125, 1235.703125, 504, 28), (\'To honor your privacy preferences, this content can only be viewed on the site it originates from.\', \'12px\', \'400\', \'16px\', 137.328125, 1279.703125, 504, 32), (\'Sign Up Today\', \'28px\', \'700\', \'28px\', 137.328125, 1338.703125, 188.828125, 31), ("This is an edition of WIRED\'s Fast Forward newsletter, a weekly dispatch from the future by Will Knight, exploring AI advances and other technology set to change our lives.", \'16px\', \'700\', \'18px\', 137.328125, 1638.6875, 270, 108), (\'Anderson doesn’t let her imaginary status get in the way of her opinions, though, and comes complete with a detailed backstory. In a wide-ranging conversation with a human interlocutor, the bot says that it has a 7-year-old son who is a fan of the New England Revolution and loves going to home games at Gillette Stadium in Massachusetts. Anderson claims to think the sport is a wonderful way for kids to stay active and make new friends.\', \'19px\', \'400\', \'28px\', 137.328125, 1334.703125, 504, 504), (\'In another conversation, two more AI characters, Jason Smith and Ashley Thompson, talk to one another about ways that Major League Soccer (MLS) might reach new audiences. Smith suggests a mobile app with an augmented reality feature showing different views of games. Thompson adds that the app could include “gamification” that lets players earn points as they watch.\', \'19px\', \'400\', \'28px\', 137.328125, 1857.703125, 504, 196), (\'The three bots are among scores of AI characters that have been developed by Fantasy, a New York company that helps businesses such as LG, Ford, Spotify, and Google dream up and test new product ideas. Fantasy calls its bots synthetic humans and says they can help clients learn about audiences, think through product concepts, and even generate new ideas, like the soccer app.\', \'19px\', \'400\', \'28px\', 137.328125, 2072.703125, 504, 224), (\'"The technology is truly incredible," says Cole Sletten, VP of digital experience at the MLS. “We’re already seeing huge value and this is just the beginning.”\', \'19px\', \'400\', \'28px\', 137.328125, 2315.703125, 504, 84), (\'Video: Fantasy\', \'11px\', \'400\', \'12.98px\', 137.328125, 2455.703125, 96.59375, 12), (\'Fantasy uses the kind of machine learning technology that powers chatbots like OpenAI’s ChatGPT and Google’s Bard to create its synthetic humans. The company gives each agent dozens of characteristics drawn from ethnographic research on real people, feeding them into commercial large language models like OpenAI’s GPT and Anthropic’s Claude. Its agents can also be set up to have knowledge of existing product lines or businesses, so they can converse about a client’s offerings.\', \'19px\', \'400\', \'28px\', 137.328125, 2529.65625, 504, 252), (\'Video: Fantasy\', \'11px\', \'400\', \'12.98px\', 137.328125, 2837.65625, 96.59375, 12), ("Fantasy then creates focus groups of both synthetic humans and real people. The participants are given a topic or a product idea to discuss, and Fantasy and its client watch the chatter. BP, an oil and gas company, asked a swarm of 50 of Fantasy’s synthetic humans to discuss ideas for smart city projects. “We\'ve gotten a really good trove of ideas,” says Roger Rohatgi, BP’s global head of design. “Whereas a human may get tired of answering questions or not want to answer that many ways, a synthetic human can keep going,” he says.", \'19px\', \'400\', \'28px\', 137.328125, 2911.609375, 504, 280), (\'Peter Smart, chief experience officer at Fantasy, says that synthetic humans have produced novel ideas for clients, and prompted real humans included in their conversations to be more creative. “It is fascinating to see novelty—genuine novelty—come out of both sides of that equation—it’s incredibly interesting,” he says.\', \'19px\', \'400\', \'28px\', 137.3281\nTITLE AND SUBTITLE:</s>\n<|assistant|>\n'
# outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)
# print(outputs[0]["generated_text"])


# In[ ]:




