# import libraries
from CompressionMethods.utils import load_data_hf, get_model_size, evaluate_heegyu_augsec, evaluate_sem_eval_2018_task_1_dataset, save_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPTQConfig
from pprint import pprint

class gptqQuantization(object):
  """
  Class that contains all the necessary functions to peform GPTQ Quantization on a model
  """
  def __init__(self, model_id, dataset_id, dataset_subsetid, model_type, is_llama=False, is_bert=True, bits=4):
    self.model_id = model_id
    self.dataset_id = dataset_id
    self.dataset_subsetid = dataset_subsetid
    self.dataset = load_data_hf(dataset_id, dataset_subsetid) 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.model_type = model_type
    self.results_gptq = {}
    self.results_gptq['method'] = 'GPTQ-quantization'

    if self.model_type == 'bert':
      self.quantization_config = GPTQConfig(bits = bits, tokenizer = self.tokenizer, dataset="c4", block_name_to_quantize='bert.encoder.layer')

    elif self.model_type == 'llama':
      self.quantization_config = GPTQConfig(bits = bits, tokenizer = self.tokenizer, dataset="c4", use_exllama=False)

    self.gptq_quantized_model = AutoModelForSequenceClassification.from_pretrained(model_id, quantization_config=self.quantization_config).to("cuda")

  def compress_model(self):
    pass

  def run_experiment(self):
    # self.compress_model = self.compress_model(self.base_model)
    self.results_gptq['size'] = get_model_size(self.gptq_quantized_model)

    if self.dataset_id == "heegyu/augesc":
        eval_results_gptq = evaluate_heegyu_augsec(self.gptq_quantized_model, self.tokenizer, self.dataset)
        self.results_gptq.update(eval_results_gptq)


    elif self.dataset_id == 'sem_eval_2018_task_1':
        eval_results_gptq = evaluate_sem_eval_2018_task_1_dataset(self.gptq_quantized_model, self.tokenizer, self.dataset)
        self.results_gptq.update(eval_results_gptq)


    print("Results of GPTQ  Quantization:")
    pprint(self.results_gptq)
    print('#'*100)

    save_model(self.gptq_quantized_model, self.tokenizer, self.model_id, "gptq-quantized")
    
    del self.gptq_quantized_model
