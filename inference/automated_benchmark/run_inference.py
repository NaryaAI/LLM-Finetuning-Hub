import subprocess
import argparse
from pathlib import Path
import click 
import sys 

@click.command()
@click.option("--huggingface_token", default="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", prompt=True, help="Your Hugging Face token")
@click.option("--huggingface_username", default="mariiaponom", prompt=True, help="Your Hugging Face username")
@click.option("--model_name", default="llama_7b_class", prompt=True, help="Model name (e.g. llama-7b-class): ")
@click.option("--model_type", default="llama", prompt=True, help="Model type (llama, flan, falcon, red_pajama):")
@click.option("--task", default="classification", prompt=True, help="Task (classification, summarization): ")
@click.option("--compute", default="a10", prompt=True, help="Compute type (a100, a10): ")
@click.option("--server", default="ray", prompt=True, help="Compute type (tgi, ray, triton, vllm): list through ', ' ")
@click.option("--lora_weights", default="./models/llama-7b-class", prompt=True, help="Lora weights")
def main(huggingface_token, huggingface_username, model_name, model_type, task, compute, server, lora_weights):
    huggingface_repo = f"{huggingface_username}/{model_name}"
    subprocess.run(["chmod", "+x", f"./script_inference.sh"])
    subprocess.run([f"./script_inference.sh", f"{huggingface_repo}", huggingface_token, 
                                        model_type, task, compute, 
                                        server, lora_weights])

if __name__ == "__main__":
    main()
