from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
import time
import numpy as np
import pandas as pd
from torch import cuda
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch
import optimum

class utils(object):
    """Utils class to help run experiments efficiently"""

    def __init__(self, model, tokenizer = None, dataset = None, encoded_dataset = None):
        """Params:
            - model(str): a path to the model directory or a model stored in HF.
            - tokenizer(str): a path to the tokenizer. By default it's assumed that the tokenizer shares the same path as the model.
            - dataset (dataset): the dataset to perform evaluations on
            - encoded_dataset (dataset): the processed dataset of dataset
            """
        super().__init__
        if tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.dataset = dataset
        self.encoded_dataset = encoded_dataset
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device)

    def preprocess_bert_data(self, examples):
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
            
    def create_encoded_dataset(self, dataset = None):
        """helper function to create the encoded dataset for BERT model"""

        # preprocess data
        try:
            if dataset:
                self.dataset = dataset
            else:
                assert self.dataset is not None, "Please provide a dataset."
            dataset = self.dataset
            encoded_dataset = dataset.map(self.preprocess_bert_data, batched=True, remove_columns=dataset['train'].column_names)
            encoded_dataset.set_format("torch")
            self.encoded_dataset = encoded_dataset
            return encoded_dataset
        except Exception as e:
            print(e)
    
    def process_tinyllama_dataset(self, dataset = None, num_samples = 1000):
        """helper function to pre-process data for LLM"""
        try:
            if dataset:
                self.dataset = dataset
            label_map = {
                "Question":0,
                "Restatement or Paraphrasing":1,
                "Reflection of feelings":2,
                "Self-disclosure":3,
                "Affirmation and Reassurance":4,
                "Providing Suggestions":5,
                "Information":6,
                "Others":7
            }
            x = []
            y_true = []
            for sample in self.dataset['test']:
                for row in sample['dialog']:
                    text = row['text']
                    label = row['strategy']
                    if label != None:
                        x.append(text)
                        y_true.append(label_map[label])
            x = x[0:num_samples]
            y_true = y_true[0:num_samples]
        except Exception as e:
            print(e)
        return x, y_true, label_map
    
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
        """Helper function to perform inferences using the model."""
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

    def convert_dict_to_labels(self, dictionary):
        return [int(dictionary[key]) for key in dictionary if key not in ['ID', 'Tweet']]
    
    def evaluate_bert_model(self, input = None, labels = None):
        """Helper function to evaluate the inference time and performance of finetuned model."""
        try:
            if not input and not labels:
                assert self.dataset is not None, "Please include a dataset to perform evaluations on." 
                input = [x['Tweet'] for x in self.dataset['validation']]
                # Iterate through the list of dictionaries and convert each one
                labels = [self.convert_dict_to_labels(data_dict) for data_dict in self.dataset['validation']]

            predictions = []
            times = []

            for input_ in input:
                inputs = self.tokenizer(input_, return_tensors = "pt", padding="max_length", truncation=True, max_length=128).to(self.device)
                st = time.time()
                with torch.no_grad():
                    output = self.model(**inputs).logits
                    times.append(time.time() - st)
                    predictions.append(output.squeeze())
            print(pd.Series(times).describe().T)
            print(self.multi_label_metrics_eval(predictions, labels))
        except ModuleNotFoundError as e:
            print("Please fine-tune the model first to perform evaluation")

    def evaluate_llm_model(self, input = None, labels = None):

        if not input and not labels:
            input, labels, label_map = self.process_tinyllama_dataset()

        y_pred = []
        times = []
        for current_x, current_y in zip(input, labels):
            inputs = self.tokenizer(current_x, return_tensors="pt").to("cuda")
            start_time = time.time()
            logits = self.model(**inputs).logits.softmax(-1)
            end_time = time.time()
            label = logits.argmax(-1).item()
            y_pred.append(label)
            times.append(end_time - start_time)
        print(accuracy_score(labels, y_pred))
        print(pd.Series(times).describe().T)

    def get_model_size(self):
        """Helper function to calculate the model size.
        """
        try:
            param_size = 0
            for param in self.model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in self.model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            model_size = (param_size + buffer_size) / 1024**2
            print('Model size: {:.3f}MB'.format(model_size))
        except ModuleNotFoundError as e:
            print("Please pass in a model")