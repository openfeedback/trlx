import trlx.trlx
from trlx.data.default_configs import default_ppo_config, TrainConfig
from trlx.utils.logging import enable_progress_bar
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    config.train.epochs = 1
    config.train.batch_size = 1
    config.model.cache_dir = '/nlp/scr/pchatain'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_name = "OpenAssistant/reward-model-deberta-v3-base"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name).to(device), AutoTokenizer.from_pretrained(reward_name)    
    reward_fn = RewardFunction(device, tokenizer, rank_model)

    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split='train', keep_in_memory=False,)
    prompts = [dict(row)["transcript"].split("\n\nAssistant:")[0] + "\n\nAssistant:" for row in dataset]
    
    enable_progress_bar()
    trainer = trlx.train(config=config, reward_fn=reward_fn, prompts=prompts) 
    
    path = 'ppo/gpt_neo_1.3B'
    trainer.save_pretrained(path)
