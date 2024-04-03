import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.nn.utils import prune
from sklearn.metrics import accuracy_score
import time
import pandas as pd
from utils import get_model_size, evaluate_heegyu_augsec, evaluate_sem_eval_2018_task_1_dataset, load_data_hf

class ModelPruner:
    """
    A class to handle pruning of transformer models for classification tasks.
    """
    def __init__(self, model_id, dataset_id=None, dataset_subset_id=None, task="classification"):
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.dataset_subset_id = dataset_subset_id
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = load_data_hf(dataset_id, dataset_subset_id) if dataset_id else None
        self.original_model_size = get_model_size(self.model)

    def apply_global_pruning(self, pruning_percentage=0.2):
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
        self.pruned_model_size = get_model_size(self.model)

    def evaluate_model(self, num_samples=1000):
        """
        Evaluate the pruned model on the given dataset.
        """
        if self.task == "classification" and self.dataset_id == "heegyu/augesc":
            results = evaluate_heegyu_augsec(self.model, self.tokenizer, self.dataset, num_samples=num_samples)
        elif self.task == "multi_label_classification" and self.dataset_id == "sem_eval_2018_task_1":
            results = evaluate_sem_eval_2018_task_1_dataset(self.model, self.tokenizer, self.dataset)
        else:
            raise ValueError("Unsupported dataset or task type.")
        self.evaluation_results = results

    def summarize_results(self):
        """
        Summarize and print the results of the pruning and evaluation.
        """
        print(f"Original Model Size: {self.original_model_size:.3f}MB")
        print(f"Pruned Model Size: {self.pruned_model_size:.3f}MB")
        print("Evaluation Results:")
        for key, value in self.evaluation_results.items():
            print(f"{key.capitalize()}: {value}")

############## sample usage ###############
# from pruning import ModelPruner

# # Define the model and dataset identifiers
# model_id = "bert-base-uncased"  # Example model identifier
# dataset_id = "heegyu/augesc"  # Example dataset identifier
# dataset_subset_id = None  # Adjust as necessary for your dataset

# # Initialize the ModelPruner
# pruner = ModelPruner(model_id=model_id, dataset_id=dataset_id, dataset_subset_id=dataset_subset_id, task="classification")

# # Apply global pruning to the model
# print("Applying global pruning...")
# pruner.apply_global_pruning(pruning_percentage=0.2)

# # Remove the pruning reparameterization to finalize the pruning process
# print("Removing pruning reparameterization...")
# pruner.remove_pruning_reparam()

# # Evaluate the pruned model on the specified dataset
# print("Evaluating the pruned model...")
# pruner.evaluate_model(num_samples=1000)

# # Summarize and print the results of the pruning and evaluation
# print("Summarizing results...")
# pruner.summarize_results()
