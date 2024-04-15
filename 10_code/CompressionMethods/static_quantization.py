from CompressionMethods.utils import load_data_hf, get_model_size, evaluate_heegyu_augsec, evaluate_sem_eval_2018_task_1_dataset, save_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from pprint import pprint
import torch

class staticQuantization(object):
    """
    Class that contains all the necessary functions to 4-bit and 8-bit quantize a model
    """
    def __init__(self, model_id, dataset_id, dataset_subsetid):
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.dataset_subsetid = dataset_subsetid
        self.dataset = load_data_hf(dataset_id, dataset_subsetid)

        self.results_4bit = {}
        self.results_4bit['method'] = '4bit-static-quantization'
        
        self.results_8bit = {}
        self.results_8bit['method'] = '8bit-static-quantization'

        quant_config_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
        self.model_4_bit = AutoModelForSequenceClassification.from_pretrained(model_id, quantization_config=quant_config_4bit, low_cpu_mem_usage=True)
        self.model_8_bit = AutoModelForSequenceClassification.from_pretrained(model_id, quantization_config=quant_config_8bit, low_cpu_mem_usage=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        

    def compress_model(self):
        pass

    def run_experiment(self):
        # self.compress_model = self.compress_model(self.base_model)
        self.results_4bit['size'] = get_model_size(self.model_4_bit)
        self.results_8bit['size'] = get_model_size(self.model_8_bit)

        if self.dataset_id == "heegyu/augesc":
            eval_results_4bit = evaluate_heegyu_augsec(self.model_4_bit, self.tokenizer, self.dataset)
            self.results_4bit.update(eval_results_4bit)
            eval_results_8bit = evaluate_heegyu_augsec(self.model_8_bit, self.tokenizer, self.dataset)
            self.results_8bit.update(eval_results_8bit)

        elif self.dataset_id == 'sem_eval_2018_task_1':
            eval_results_4bit = evaluate_sem_eval_2018_task_1_dataset(self.model_4_bit, self.tokenizer, self.dataset)
            self.results_4bit.update(eval_results_4bit)
            
            eval_results_8bit = evaluate_sem_eval_2018_task_1_dataset(self.model_8_bit, self.tokenizer, self.dataset)
            self.results_8bit.update(eval_results_8bit)

        print("Results of 4-bit Static Quantization:")
        pprint(self.results_4bit)
        print('#'*100)

        print("Results of 8-bit Static Quantization:")
        pprint(self.results_8bit)
        print('#'*100)

        save_model(self.model_4_bit, self.tokenizer, self.model_id, "4-bit-static-quantized")
        save_model(self.model_8_bit, self.tokenizer, self.model_id, "8-bit-static-quantized")

        del self.model_4_bit
        del self.model_8_bit


        



