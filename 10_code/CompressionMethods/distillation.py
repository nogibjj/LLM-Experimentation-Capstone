from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EvalPrediction
from torch import cuda
import torch
import torch.nn.functional as F
from CompressionMethods.utils import load_data_hf, preprocess_bert_data, compute_metrics, get_model_size, evaluate_sem_eval_2018_task_1_dataset, save_model
import time
from pprint import pprint

class KnowledgeDistillationTrainer(Trainer):
  def __init__(self, *args, teacher_model=None, temperature = 2.0, alpha = 0.5, **kwargs):
    super().__init__(*args, **kwargs)
    self.teacher_model = teacher_model
    self.temperature = temperature
    self.alpha = alpha

  def compute_loss(self, model, inputs, return_outputs=False):
    """Function to compute the loss of distilled model
    Params:
        model: student model
        inputs (list): a list of inputs
        return_outputs: returns a tuple of the loss and model outputs if True, otherwise it returns the loss only
    Returns:
        (loss, output) if return_outputs is True, else it returns the loss
    """
    #Extract cross-entropy loss and logits from student
    outputs_student = model(**inputs)
    loss_ce = outputs_student.loss
    logits_student = outputs_student.logits

    # Extract logits from teacher
    outputs_teacher = self.teacher_model(**inputs)
    logits_teacher = outputs_teacher.logits

     #Computing distillation loss by Softening probabilities
    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
    #The reduction=batchmean argument in nn.KLDivLoss() specifies that we average the losses over the batch dimension.
    loss_kd = self.temperature ** 2 * loss_fct(
                F.log_softmax(logits_student / self.temperature, dim=-1),
                F.softmax(logits_teacher / self.temperature, dim=-1))

    # Return weighted student loss
    loss = self.alpha * loss_ce + (1. - self.alpha) * loss_kd
    return (loss, outputs_student) if return_outputs else loss

class DistillationModule(object):
    """Class for creating a distilled model for BERT"""

    def __init__(self, teacher_model_id, student_model_id, dataset, sub_dataset = None):
        """Establishing the parameters for the distilled model
        Attributes:
            - self.device: device which the experiments are run on
            -teacher_model (str): the teacher model on HF used for distillation. For our experimentations, we used fine-tuned BERT model
            -student_model (str): the student model on HF used for distillation. For our experimentations, we used DistilBERT
            -dataset (str): the encoded / preprocessed dataset to train the model on
            -sub_dataset (str):
        """
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.teacher_model_id = teacher_model_id
        self.student_model_id = student_model_id
        self.dataset = dataset
        self.sub_dataset = sub_dataset
        self.tokenizer = None
        self.distilled_model = None
        self.results_distillation = {}
        self.results_distillation['method'] = 'Distillation (BERT Only)'

    def get_device(self):
        """Obtaining the device where experiments are performed"""
        print("Using device: ", self.device)

    def perform_distillation(self, preprocessing_func = None, output_dir = "./bert-distilled", learning_rate = 2e-5, batch_size = 8, num_epochs = 5, metric_name = "f1", weight_decay = 0.01, alpha = 0.5, temperature = 2.0, num_labels = 11):
       """Helper function to perform knowledge distillation.
          Params:
            preprocessing_func (method): the function used to preprocess the dataset
            output_dir (str): the output directory with distilled model
            **args for the training process
          Returns:
            distilled_model: the distilled model
       """
       
       student_training_args = TrainingArguments(
          output_dir = output_dir,
          evaluation_strategy = "epoch",
          save_strategy = "epoch",
          learning_rate = learning_rate,
          per_device_train_batch_size=batch_size,
          per_device_eval_batch_size=batch_size,
          num_train_epochs=num_epochs,
          weight_decay=weight_decay,
          load_best_model_at_end=True,
          metric_for_best_model=metric_name
       )

       student_model = AutoModelForSequenceClassification.from_pretrained(self.student_model_id, num_labels = num_labels)

       student_model = student_model.to(self.device)

       student_tokenizer = AutoTokenizer.from_pretrained(self.student_model_id)

       self.tokenizer = student_tokenizer

       # Use the fine-tuned model as the teacher
       teacher_model = AutoModelForSequenceClassification.from_pretrained(self.teacher_model_id, num_labels = num_labels)

       teacher_model = teacher_model.to(self.device)

       dataset = load_data_hf(self.dataset, self.sub_dataset)

       if not preprocessing_func:
          preprocessing_func = preprocess_bert_data
        
       encoded_dataset = dataset.map(preprocessing_func, fn_kwargs={'tokenizer' : student_tokenizer}, batched=True, remove_columns=dataset['train'].column_names)
       encoded_dataset.set_format("torch")

       distilled_model = KnowledgeDistillationTrainer(student_model, teacher_model= teacher_model,
                                                        args = student_training_args, train_dataset=encoded_dataset["train"],
                                                        eval_dataset=encoded_dataset["validation"],
                                                        compute_metrics=compute_metrics, 
                                                        tokenizer = student_tokenizer)
       start_time = time.time()
       distilled_model.train()
       end_time = time.time()

       print("Training time: ", end_time - start_time)

       distilled_model.save_model(output_dir)

       distilled_model = AutoModelForSequenceClassification.from_pretrained(output_dir, num_labels = num_labels).to(self.device)

       self.distilled_model = distilled_model
       save_model(self.distilled_model, self.tokenizer, self.teacher_model_id, "distilled")

       return distilled_model
    
    def calculate_model_size(self):
       return get_model_size(self.distilled_model)
    
    def evaluate_bert_model(self):
       tokenizer = AutoTokenizer.from_pretrained(self.student_model_id)
       dataset = load_data_hf(self.dataset, self.sub_dataset)
       return evaluate_sem_eval_2018_task_1_dataset(self.distilled_model, tokenizer, dataset, self.device)

    def run_experiment(self):
       distilled_model = self.perform_distillation()
       self.results_distillation['size'] = get_model_size(distilled_model)

       dataset = load_data_hf(self.dataset, self.sub_dataset)
       eval_results_distillation = evaluate_sem_eval_2018_task_1_dataset(distilled_model, self.tokenizer, dataset)
       self.results_distillation.update(eval_results_distillation)

       print("Results of Distillation:")
       pprint(self.results_distillation)
       print('#'*100)
