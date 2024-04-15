import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.nn.utils import prune
from sklearn.metrics import accuracy_score
import time
import pandas as pd
from CompressionMethods.utils import get_model_size, evaluate_heegyu_augsec, evaluate_sem_eval_2018_task_1_dataset, save_model, load_data_hf

class PruneModel:
    """
    A class to handle pruning of transformer models for classification tasks.
    """
    def __init__(self, model_id, dataset_id=None, dataset_subsetid=None):
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.dataset_subsetid = dataset_subsetid
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = load_data_hf(dataset_id, dataset_subsetid) if dataset_id else None
        self.original_model_size = get_model_size(self.model)
        self.results = {} 
        self.results['method'] = 'Pruning'

    def apply_global_pruning(self, pruning_percentage=0.2): # pruning percentage subject to change
        """
        Apply global L1 unstructured pruning to the model.
        """
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_percentage,
        )

    def remove_pruning_reparam(self):
        """
        Remove the pruning reparameterization to finalize the pruning process.
        """
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                prune.remove(module, "weight")
        self.results['size'] = get_model_size(self.model)

    def evaluate_model(self):
        """
        Evaluate the pruned model on the given dataset.
        """
        if self.dataset_id == "heegyu/augesc":
            eval_results = evaluate_heegyu_augsec(self.model, self.tokenizer, self.dataset)

        elif self.dataset_id == 'sem_eval_2018_task_1':
            eval_results = evaluate_sem_eval_2018_task_1_dataset(self.model, self.tokenizer, self.dataset)
        else:
            raise ValueError("Unsupported dataset ID for evaluation.")
        
        self.results.update(eval_results)  

    def run_experiment(self):
        self.apply_global_pruning(pruning_percentage=0.2)
        self.remove_pruning_reparam()
        self.evaluate_model()
        save_model(self.model, self.tokenizer, self.model_id, "pruned") 
        self.summarize_results()

    def summarize_results(self):
        """
        Summarize and print the results of the pruning and evaluation.
        """
        print("Results of Pruning:")
        print(self.results)
        print('#'*100)
