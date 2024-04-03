import torch
from torch import cuda
from torchao.quantization import quant_api
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import time
from torch.nn.utils import prune
import pandas as pd
import onnx
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import warnings
import numpy as np
warnings.filterwarnings('ignore')

class ONNXconverter(object):
    def __init__(self, model_checkpoint,model_load_method="huggingface",device="cpu",op_version = 14, architecture="transformer"):
        self.model_checkpoint = model_checkpoint
        self.model_load_method = model_load_method
        
        # initalize device as GPU if and only if the user select GPU and cuda is available
        if device=="gpu" and cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.op_version = op_version
        self.architecture = architecture
    
    def onnx_convert_classification(self, tokenizer_load_method, sample_input, model_checkpoint=None, tokenizer=None, onnx_path=None, device="cpu"):
        """
        Converts a pre-trained classification or language model to ONNX format for optimized inference.
        
        This function supports converting models based on BERT or GPT architecture from HuggingFace's Transformers
        library or custom sources. It requires specifying loading methods for both the model and tokenizer, handling
        them according to the provided sources (HuggingFace, local, or environment).

        Parameters:
        - model_checkpoint (str, optional): Identifier for the model, which can be a HuggingFace model ID, local path, 
        or already loaded model in the environment. Used to load the model when not provided explicitly.
        - tokenizer (str, optional): Identifier for the tokenizer, similar to `model_checkpoint`. Used for processing 
        `sample_input` to match model's expected input format.
        - model_load_method (str): Specifies how to load the model. Acceptable values: ['huggingface', 'local', 'env'].
        - tokenizer_load_method (str): Specifies how to load the tokenizer. Acceptable values: ['huggingface', 'local', 'env'].
        - sample_input (any): Example input for the model to define the input tensor shape for ONNX.
        - onnx_path (str, optional): File path where the ONNX model will be saved. If not provided, the model is not saved.
        - device (str): Computation device to use ('cpu' or 'cuda') for model conversion.
        - op_ver (int): Specifies the ONNX Operator Set Version to use.
        - architecture (str): The architecture of the model to convert ('bert' or 'gpt').

        Returns:
        - The converted ONNX model or None if the conversion fails.
        """
        model_load_method = self.model_load_method
        model_checkpoint = self.model_checkpoint
        op_ver = self.op_version
        architecture = self.architecture
        
        # Load the model based on the specified architecture and device
        if model_load_method in ["huggingface", "local"]:
            if architecture == "bert":
                model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).eval().to(device)
            elif architecture == "gpt":
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint).eval().to(device)
        else:
            print("Unsupported model loading method.")
            return None

        # Load the tokenizer
        if tokenizer_load_method in ["huggingface", "local"]:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer if tokenizer else model_checkpoint)
        else:
            print("Unsupported tokenizer loading method.")
            return None

        # Check if both model and tokenizer are successfully loaded
        if model is None or tokenizer is None:
            print("Error: Model or tokenizer failed to load.")
            return None

        # Prepare the sample input for ONNX export by tokenizing it according to the model's needs
        inputs = tokenizer(sample_input, return_tensors="pt", padding=True, truncation=True).to(device)
        # Determine input arguments based on model architecture
        input_args = (inputs['input_ids'],) if architecture == "gpt" else (inputs['input_ids'], inputs['attention_mask'])

        # Perform the ONNX export
        torch.onnx.export(model,
                        args=input_args,
                        f=onnx_path,
                        input_names=['input_ids'] if architecture == "gpt" else ['input_ids', 'attention_mask'],
                        output_names=['logits'],
                        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                        'attention_mask': {0: 'batch_size', 1: 'sequence_length'} if architecture != "gpt" else {},
                                        'logits': {0: 'batch_size'}},
                        opset_version=op_ver)

        # Return the model after successful conversion
        return model
    

"""

Following parts are designed for the experiment reproduction. The ideal input is from huggingface
The comprehensive method will 1. load the original model 2. evaluate the original model 3. convert to onnx 4. evaluate onnx model
The comprehensive method will not return onnx model, so please use previous method for convert

"""

# calculate softmax for onnx since onnx doesn't have built in softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# evaluate original model

def evaluate_model_performance(self, model, tokenizer, x = None, y_true = None, dataloader = None, input_type = "array", architecture = "transformer", device="cpu"):

    # calculate model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_size = (param_size + buffer_size) / 1024**2
    print('Base Model size: {:.3f}MB'.format(model_size))
    if architecture == "transformer":
      if input_type == "array":
        # inference part - measure speed and accuracy
        y_pred = []
        times = []
        for current_x, current_y in zip(x, y_true):
            inputs = tokenizer(current_x, return_tensors="pt").to(device)
            start_time = time.time()
            logits = model(**inputs).logits.softmax(-1)
            end_time = time.time()
            label = logits.argmax(-1).item()
            y_pred.append(label)
            times.append(end_time - start_time)

      elif input_type == "dataloader":
        for batch in dataloader:
          inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
          start_time = time.time()
          logits = model(**inputs).logits.softmax(-1)
          end_time = time.time()
          label = logits.argmax(-1).item()
          y_pred.append(label)
          times.append(end_time - start_time)
      else:
        print("Wrong input type")
        return None

      acc = accuracy_score(y_true, y_pred)
      print(acc)
      print(pd.Series(times).describe().T)
      return model_size, sum(times) / len(times), acc

# evaluate onnx model

def evaluate_onnx_model(self, ort_session, tokenizer, x=None, y_true=None, dataloader=None, input_type="array", device="cpu"):
    predictions = []
    times = []

    # For handling inputs as arrays
    if input_type == "array":
        for text in x:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            onnx_inputs = {'input_ids': np.ascontiguousarray(inputs['input_ids'].cpu().numpy()),
                           'attention_mask': np.ascontiguousarray(inputs['attention_mask'].cpu().numpy())}

            start_time = time.time()
            logits = ort_session.run(None, onnx_inputs)[0]
            end_time = time.time()
            probs = softmax(logits, axis=-1)
            prediction = np.argmax(probs, axis=-1).item()
            predictions.append(prediction)
            times.append(end_time - start_time)

    # For handling inputs as a DataLoader
    elif input_type == "dataloader" and dataloader is not None:
        for batch in dataloader:
            inputs = tokenizer(batch['texts'], return_tensors="pt", padding=True, truncation=True).to(device)
            onnx_inputs = {'input_ids': np.ascontiguousarray(inputs['input_ids'].cpu().numpy()),
                           'attention_mask': np.ascontiguousarray(inputs['attention_mask'].cpu().numpy())}

            start_time = time.time()
            logits = ort_session.run(None, onnx_inputs)[0]
            end_time = time.time()
            probs = softmax(logits, axis=-1)
            prediction = np.argmax(probs, axis=-1).flatten().tolist()
            predictions.extend(prediction)
            times.append(end_time - start_time)

    else:
        print("Wrong input type")
        return None

    predictions = np.array(predictions)
    acc = accuracy_score(y_true, predictions)
    inference_speed = sum(times) / len(times)
    print(f'Accuracy: {acc}')
    print(pd.Series(times).describe().T)
    return inference_speed, acc

def convert_and_evaluate(self, tokenizer_load_method, sample_input, model_checkpoint=None, tokenizer_checkpoint=None, onnx_path=None, device="cpu"):
    
    model_load_method = self.model_load_method
    model_checkpoint = self.model_checkpoint
    op_ver = self.op_version
    architecture = self.architecture
    
    if model_load_method == "huggingface":
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).eval().to(device)


    if tokenizer_load_method == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)


    original_size, original_inference_speed, original_accuracy = evaluate_model_performance(model, tokenizer, x, y, device)
    print(f"Original Model - Size: {original_size} Inference Speed: {original_inference_speed}s, Accuracy: {original_accuracy}")
    start_time = time.time()
    model = self.onnx_convert_classification(self, tokenizer_load_method, sample_input, model_checkpoint=None, tokenizer=None, onnx_path=None, device="cpu")
    end_time = time.time()
    print(f"ONNX converting time is {end_time - start_time}s")
    
    # intialize ort session
    sess_options = ort.SessionOptions()
    # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = onnx_path
    ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CUDAExecutionProvider'])
    # Evaluate the ONNX model

    onnx_inference_speed, onnx_accuracy = evaluate_onnx_model(ort_session, tokenizer, x, y, device = device)
    print(f"ONNX Model - Inference Speed: {onnx_inference_speed}s, Accuracy: {onnx_accuracy}")

    return {
        "original": {"inference_speed": original_inference_speed, "accuracy": original_accuracy},
        "onnx": {"inference_speed": onnx_inference_speed, "accuracy": onnx_accuracy}
    }
