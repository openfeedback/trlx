import trlx.trlx
from trlx.data.default_configs import default_ppo_config, TrainConfig
from trlx.utils.logging import enable_progress_bar
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from typing import Iterator, TypeVar

T = TypeVar("T")


def get_superhf_prompts(dataset_name: str, split: str = "train") -> list[str]:
    """
    Get a list of prompts from a dataset.
    Args:
        dataset_name: The name of the dataset to load. One of:
            - 'anthropic-red-team'
        split: The split of the dataset to load.
    Returns:
        A list of prompts.
    """
    # Load the appropriate dataset then convert to a list of prompts
    prompts: list[str] = []
    if dataset_name == "anthropic-red-team":
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="red-team-attempts",
            split=split,
            keep_in_memory=False,
        )
        prompts.extend(
            [
                dict(row)["transcript"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                for row in dataset
            ]
        )
    elif dataset_name == "openai/webgpt_comparisons":
        dataset = load_dataset(
            dataset_name,
            split=split,
            keep_in_memory=False,
        )
        prompts.extend(
            [
                "\n\nHuman: " + row["question"]["full_text"] + "\n\nAssistant:"
                for row in dataset
            ]
        )
    elif dataset_name == "anthropic-harmless-base":
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="harmless-base",
            split=split,
            keep_in_memory=False,
        )
        prompts.extend(
            [
                row["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                for row in dataset
            ]
        )
    elif dataset_name == "anthropic-helpful-base":
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="helpful-base",
            split=split,
            keep_in_memory=False,
        )
        prompts.extend(
            [
                row["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                for row in dataset
            ]
        )
    elif dataset_name == "mock":
        prompts.extend(
            [
                f"{i}\n\nHuman: ...\n\nAssistant: Sphinx of black quartz, judge my vow."
                for i in range(50000)
            ]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return prompts

class RewardFunction:
    def __init__(self, device, reward_model_tokenizer, reward_model):
        self.device = device
        self.tokenizer = reward_model_tokenizer
        self.model = reward_model
        
    def __call__(self, prompts, outputs, **kwargs):
 
        inputs = self.tokenizer(
            prompts, 
            outputs, 
            truncation=True,
            padding=True,
            return_tensors='pt',
        ).to(self.device)
        
        return self.model(**inputs).logits.flatten().tolist()


if __name__ == '__main__':
    config = default_ppo_config()
    config.model.model_path = 'EleutherAI/gpt-neo-1.3B'
    config.model.num_layers_unfrozen = 4
    config.train.seq_length = 256
    config.train.epochs = 4
    config.train.batch_size = 1
    config.model.cache_dir = '/nlp/scr/pchatain'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_name = "OpenAssistant/reward-model-deberta-v3-base"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name).to(device), AutoTokenizer.from_pretrained(reward_name)    
    reward_fn = RewardFunction(device, tokenizer, rank_model)

    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split='train', keep_in_memory=False,)
    prompts = []
    for dataset_name in ("anthropic-red-team", "openai/webgpt_comparisons", "anthropic-helpful-base"):
        prompts += get_superhf_prompts(dataset_name)

    # prompts = [dict(row)["transcript"].split("\n\nAssistant:")[0] + "\n\nAssistant:" for row in dataset]
    
    enable_progress_bar()
    trainer = trlx.train(config=config, reward_fn=reward_fn, prompts=prompts) 
    
    path = 'ppo/gpt_neo_1.3B'
    trainer.save_pretrained(path)
