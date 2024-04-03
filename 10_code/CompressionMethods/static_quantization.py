from CompressionMethods.utils import load_data_hf, get_model_size, evaluate_heegyu_augsec, evaluate_sem_eval_2018_task_1_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pprint import pprint
class staticQuantization(object):
    """
    Class that contains all the necessary functions to 4-bit and 8-bit quantize a model
    """
    def __init__(self, model_id, dataset_id, dataset_subsetid):
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.dataset_subsetid = dataset_subsetid

        self.results_4bit = {}
        self.results_4bit['method'] = '4bit-static-quantization'
        
        self.results_8bit = {}
        self.results_8bit['method'] = '8bit-static-quantization'

        self.model_4_bit = AutoModelForSequenceClassification.from_pretrained(model_id, load_in_4bit=True)
        self.model_8_bit = AutoModelForSequenceClassification.from_pretrained(model_id, load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.dataset = load_data_hf(dataset_id, dataset_subsetid)

    def quantize_model(self):
        pass

    def run_experiment(self):
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

        



