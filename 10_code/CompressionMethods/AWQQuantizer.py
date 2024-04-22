# import libraries
from CompressionMethods.utils import load_data_hf, get_model_size, evaluate_heegyu_augsec, evaluate_sem_eval_2018_task_1_dataset, save_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AwqConfig 
from awq import AutoAWQForCausalLM
from pprint import pprint

class AWQQuantization(object):
  """
  Class that contains all the necessary functions to peform AWQ Quantization on a model
  """
  def __init__(self, model_id, dataset_id, dataset_subsetid, model_type, bits=4, num_labels = 8):
    self.model_id = model_id
    self.dataset_id = dataset_id
    self.dataset_subsetid = dataset_subsetid
    self.dataset = load_data_hf(dataset_id, dataset_subsetid) 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    self.model_type = model_type
    self.results_awq = {}
    self.results_awq['method'] = 'AWQ-quantization'
    self.device = 'cuda'

    assert self.model_type == 'llama', f"AWQ can not be applied to {self.model_type} model."

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": bits, "version":"GEMM"}
    
    # Load model
    model = AutoAWQForCausalLM.from_pretrained(self.model_id)

    # Quantize
    model.quantize(self.tokenizer, quant_config=quant_config)

    # modify the config file so that it is compatible with transformers integration
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    model.config.quantization_config = quantization_config

    save_model(model, self.tokenizer, self.model_id, "awq-quantized")

    model_name = model_id.split('/')[-1]
    model_path = f"models/{model_name}/awq-quantized"

    self.awq_quantized_model =  AutoModelForSequenceClassification.from_pretrained(model_path, device_map = "auto", num_labels = num_labels).to("cuda")

  def compress_model(self):
    pass

  def run_experiment(self):
    # self.compress_model = self.compress_model(self.base_model)
    self.results_awq['size'] = get_model_size(self.awq_quantized_model)

    if self.dataset_id == "heegyu/augesc":
        eval_results_awq = evaluate_heegyu_augsec(self.awq_quantized_model, self.tokenizer, self.dataset)
        self.results_awq.update(eval_results_awq)


    elif self.dataset_id == 'sem_eval_2018_task_1':
        eval_results_awq = evaluate_sem_eval_2018_task_1_dataset(self.awq_quantized_model, self.tokenizer, self.dataset)
        self.results_awq.update(eval_results_awq)


    print("Results of AWQ Quantization:")
    pprint(self.results_awq)
    print('#'*100)

    del self.awq_quantized_model
