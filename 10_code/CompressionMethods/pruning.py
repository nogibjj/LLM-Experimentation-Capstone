import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import time
from utils import (
    utils,
)  # Import the utility class you provided for handling data preprocessing and metrics


class PruneModel(object):
    """Class to handle pruning and fine-tuning of transformer models."""

    def __init__(self, model_name, dataset_name, subtask_name):
        """
        Initializes the PruneModel class with model, dataset, and utility objects.

        Parameters:
            model_name (str): The name of the model to be pruned.
            dataset_name (str): The name of the dataset to be used.
            subtask_name (str): The specific subtask of the dataset to be used.
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.subtask_name = subtask_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, problem_type="multi_label_classification"
        ).to(self.device)
        self.utils = utils(
            model_name=model_name
        )  # Initialize utils for dataset processing and metrics
        self.dataset = self.utils.create_encoded_dataset(
            load_dataset(dataset_name, subtask_name)
        )

    def prune_model(self, amount=0.2):
        """
        Applies pruning to the model to reduce its size or computation.

        Parameters:
            amount (float): The proportion of weights to prune away. Default is 0.2 (20%).
        """
        parameters_to_prune = (
            (self.model.bert.embeddings.word_embeddings, "weight"),
            (self.model.bert.encoder.layer[0].attention.self.query, "weight"),
            (self.model.bert.encoder.layer[0].attention.self.key, "weight"),
            (self.model.bert.encoder.layer[0].attention.self.value, "weight"),
            (self.model.bert.encoder.layer[0].attention.output.dense, "weight"),
            (self.model.bert.encoder.layer[0].intermediate.dense, "weight"),
            (self.model.bert.encoder.layer[0].output.dense, "weight"),
            # Add more layers here as needed for pruning
        )
        torch.nn.utils.prune.global_unstructured(
            parameters_to_prune,
            pruning_method=torch.nn.utils.prune.L1Unstructured,
            amount=amount,  # Prune 20% of the weights
        )
        # Remove the pruning reparameterization for a cleaner model
        for module, name in parameters_to_prune:
            torch.nn.utils.prune.remove(module, name)
        print("Pruning complete.")

    def fine_tune_pruned_model(
        self, output_dir, num_train_epochs=3, per_device_train_batch_size=8
    ):
        """
        Fine-tunes the pruned model on the given dataset.

        Parameters:
            output_dir (str): Directory to save the fine-tuned model.
            num_train_epochs (int): Number of epochs for fine-tuning.
            per_device_train_batch_size (int): Batch size per device for training.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.utils.compute_metrics,  # Utilize compute_metrics from utils for evaluation
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
        )
        trainer.train()
        self.model.save_pretrained(output_dir)


# Example usage
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    dataset_name = "sem_eval_2018_task_1"
    subtask_name = "subtask5.english"
    output_dir = "./bert-pruned"

    pruner = PruneModel(model_name, dataset_name, subtask_name)
    pruner.prune_model(amount=0.2)  # Prune 20% of the weights
    pruner.fine_tune_pruned_model(
        output_dir, num_train_epochs=3
    )  # Fine-tune and save the pruned model
