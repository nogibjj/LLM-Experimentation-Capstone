# import libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch import cuda
import accelerate
from awq import AutoAWQForCausalLM

class AWQQuantizer(object):
  """Class to quantize a LLM using AWQ method."""

  def __init__(self, model_id):
    """
    Params:
      model_id (str): the model name from HuggingFace. Note: some models need prior login into hugginface_hub.
      device (str): device on which the model is stored
    """
    self.model_id = model_id
    self.device = 'cuda' if cuda.is_available() else 'cpu'
    self.dataset = None
    self.tokenizer = None
    self.quantized_model = None
    self.quantized_model_path = None

  def load_data_hf(self, dataset_id = None, dataset_subid = None):
    """Helper function to load data.
    Params:
      dataset_id (str): the dataset stored from HuggingFace. If this is none, you can specify your custom dataset in the tokenizer class.
      dataset_subid (str): the specific sub dataset within the corpus of data in huggingface
    Returns:
      dataset (tensor): dataset loaded from HuggingFace

    """
    if not dataset_subid:
      dataset = load_dataset(dataset_id)
    else:
      dataset = load_dataset(dataset_id, dataset_subid)
    self.dataset = dataset
    return dataset

  def quantize(self, dataset, tokenizer_id = None, bits = 4, quantized_path = None):
    """Helper function to quantize the model.
    Params:
      dataset (list): list of strings that are used for quantization. If you use a dataset from HuggingFace, make sure to select appropriate dataset ('test, validation etc.)
      tokenizer (tokenizer): tokenizer used to process the text)
      bits (int): the bit-size for quantization
      quantized_path (str) = the path where you want to save the model in your workspace
    Returns:
      model (model): quantized model
    """
    # Assign tokenizer
    if tokenizer_id is None:
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    else:
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    # Initialize quantization configurations
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version":"GEMM"}

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(self.model_id)
    tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # modify the config file so that it is compatible with transformers integration
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    # the pretrained transformers model is stored in the model attribute + we need to pass a dict
    model.config.quantization_config = quantization_config
    # a second solution would be to use Autoconfig and push to hub (what we do at llm-awq)


    # save model weights
    quant_path = './awq-tinyllama-classification'
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    quantized_model = AutoModelForSequenceClassification.from_pretrained(quant_path, device_map = "auto", num_labels = 8)

    return model

  def push_to_hub(self, model, model_path):
    """Helper function to push the model onto HuggingFace. Make sure that
    the model_path is a valid repository in HuggingFace. Also make sure that you're logged in to hugginface-cli
    using a token with write access

    Params:
      model(model): model you want to push to HF
      model_path(str): the path / repository path in HF

    Returns: None

    """
    model.push_to_hub(model_path)

    # will load the model uploaded to hf
    self.quantized_model_path = model_path

  def load_model(self, classification = True, is_llama = False, is_bert = False):
    """Load the model for inference
    Params:
      classification (Boolean): will load an AutoSequenceForClassification for classification tasks
      is_llama (Boolean): set to True if quantizing a llama model
      is_bert (Boolean): set to True if quantizing a BERT model
    Returns:
      model (model): returns the quantized model for inference

    """
    if classification:
      model = AutoModelForSequenceClassification.from_pretrained(self.quantized_model_path, device_map = 'auto')
    else:
      model = AutoModelForCausalLM.from_pretrained(self.quantized_model_path, device_map="auto")

    # setting the correct configurations
    if not is_llama:
      model.config.quantization_config.use_exllama = False

    if is_bert:
      model.config.quantization_config.block_name_to_quantize = 'bert.encoder.layer'

    self.quantized_model = model

    self.calculate_model_size(model)

    model = model.to(self.device)

    return model

  def get_tokenizer(self):
    return self.tokenizer

  def get_dataset(self):
    return self.dataset