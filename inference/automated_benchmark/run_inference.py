import subprocess
import argparse
from pathlib import Path
import click 
import sys 
import typer

def main():
  
    huggingface_token = typer.prompt("HuggingFace token")
    huggingface_repo = typer.prompt("HuggingFace repository")
    model_type = typer.prompt("Model type")
    task = typer.prompt("Task")
    server = typer.prompt("Server")
    path_to_lora_weights = typer.prompt("Path to lora weights")

    subprocess.run(["chmod", "+x", f"./script_inference.sh"])
    subprocess.run([f"./script_inference.sh", f"{huggingface_repo}", huggingface_token, 
                                        model_type, task, server, path_to_lora_weights])

if __name__ == "__main__":
    typer.run(main)
