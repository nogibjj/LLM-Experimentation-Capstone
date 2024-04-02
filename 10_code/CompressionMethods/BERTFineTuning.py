from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset
import time
import numpy as np
import pandas as pd
from torch import cuda
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch
import os

class BERTFineTuning(object):
    """Class to help fine-tune the BERT model on a emotions 
    tweet multilabel classification task"""

    def __init__(self, model, output_path = './bert-finetuned'):
        """initializing the class"""
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.output_path = output_path
        self.finetuned_model = None
    
    def get_device(self):
        """Obtaining the device where experiments are performed"""
        print("Using device: ", self.device)
    
    def preprocess_data(self, examples):
        """helper function to preprocess the data for the emotions classifier dataset
        Note: this is only suited for the Tweets dataset. Do not use this for any other datasets.
        """
        # prepare the labels dataset for inference
        labels = [label for label in examples if label not in ['ID', 'Tweet']]
        id2label = {idx:label for idx, label in enumerate(labels)}
        label2id = {label:idx for idx, label in enumerate(labels)}

        # take a batch of texts
        text = examples["Tweet"]
        # encode them
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
        
        return encoding
            
    def create_encoded_dataset(self, dataset):
        """helper function to create the encoded dataset"""

        # preprocess data
        encoded_dataset = dataset.map(self.preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
        encoded_dataset.set_format("torch")

        return encoded_dataset
    
    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
        return metrics

    def multi_label_metrics_eval(self, predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        #sigmoid = torch.nn.Sigmoid()
        #probs = [sigmoid(tensor) for tensor in predictions]
        # next, use threshold to turn them into integer predictions
        y_pred = []
        for inner_list in predictions:
            new_inner_list = [1 if value > threshold else 0 for value in list(inner_list)]
            y_pred.append(new_inner_list)

        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
        return metrics
    
    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        result = self.multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids)
        return result
    
    def finetune(self, dataset = None, encoded_dataset = None, labels = None, learning_rate = 2e-5, batch_size = 8, num_train_epochs = 5, weight_decay = 0.01, eval_metric = "f1"):
        """helper function to finetune the model
            Params:
                ccc
            Returns:
                finetuned model
        """

        # load dataset
        if dataset is None:
            dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
            encoded_dataset = self.create_encoded_dataset(dataset)

            # prepare the labels dataset for inference
            labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
        
        id2label = {idx:label for idx, label in enumerate(labels)}
        label2id = {label:idx for idx, label in enumerate(labels)}

        model = AutoModelForSequenceClassification.from_pretrained(self.model, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)
        
        training_args = TrainingArguments(
            self.output_path,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate = learning_rate,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            num_train_epochs = num_train_epochs,
            weight_decay = weight_decay,
            load_best_model_at_end = True,
            metric_for_best_model = eval_metric
        )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        start_time = time.time()
        trainer.train()
        end_time = time.time()

        print(f"Training time: {end_time - start_time}")

        # save the best model
        trainer.save_model(self.output_path)
        self.finetuned_model = AutoModelForSequenceClassification.from_pretrained(self.output_path, problem_type = "multi_label_classification", num_labels = len(labels), id2label = id2label, label2id = label2id).to(self.device)

        return self.finetuned_model

    def evaluate_model(self, input, labels):
        """Helper function to evaluate the inference time and performance of finetuned model."""
        
        try:
            predictions = []
            times = []

            if not self.finetuned_model:
                if os.path.isdir(self.output_path):
                    self.finetuned_model = AutoModelForSequenceClassification.from_pretrained(self.output_path, problem_type = "multi_label_classification")
                    self.finetuned_model = self.finetuned_model.to(self.device)
                else:
                    raise ModuleNotFoundError

            for input_ in input:
                inputs = self.tokenizer(input_, return_tensors = "pt", padding="max_length", truncation=True, max_length=128).to(self.device)
                st = time.time()
                with torch.no_grad():
                    output = self.finetuned_model(**inputs).logits
                    times.append(time.time() - st)
                    predictions.append(output.squeeze())
            print(pd.Series(times).describe().T)
            print(self.multi_label_metrics_eval(predictions, labels))
        except ModuleNotFoundError as e:
            print("Please fine-tune the model first to perform evaluation")

    def get_model_size(self):
        """Helper function to calculate the model size of fine-tuned model."""
        try:
            if not self.finetuned_model:
                if os.path.isdir(self.output_path):
                    self.finetuned_model = AutoModelForSequenceClassification.from_pretrained(self.output_path, problem_type = "multi_label_classification")
                    self.finetuned_model = self.finetuned_model.to(self.device)
                else:
                    raise ModuleNotFoundError
            
            param_size = 0
            for param in self.finetuned_model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in self.finetuned_model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            model_size = (param_size + buffer_size) / 1024**2
            print('Model size: {:.3f}MB'.format(model_size))
        except ModuleNotFoundError as e:
            print("Please fine-tune the model first to get its size")
