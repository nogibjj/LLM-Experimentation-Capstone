from CompressionMethods import static_quantization, distillation, pruning, GPTQQuantizer
from dataclasses import dataclass, field
from typing import Optional
from dataclasses import field
import pandas as pd
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    model: str = field(default="adjohn1313/bert-base-finetuned", metadata={"help": "The model location (huggingface or local) to use for experimentation"})
    model_type: str = field(default="bert", metadata={"help":"the type of model being experimented on"})
    dataset: str = field(default="sem_eval_2018_task_1", metadata={"help": "The dataset used to assess model's inference speed and accuracy."})
    dataset_subtask: Optional[str] = field(default="subtask5.english", metadata={"help": "Path to save all the compressed model, if required"})
    student_model: str = field(default="distilbert/distilbert-base-uncased", metadata={"help": "The student model location (huggingface or local) to use for experimentation. Applies to distillation only"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print("Running with the following arguments:")
print(f"--model: {script_args.model}")
print(f"--model_type: {script_args.model_type}")
print(f"--dataset: {script_args.dataset}")
print(f"--dataset_subtask: {script_args.dataset_subtask}")

staticQuantizationObject = static_quantization.staticQuantization(model_id=script_args.model, dataset_id=script_args.dataset, dataset_subsetid=script_args.dataset_subtask)
staticQuantizationObject.run_experiment()

distillationObject = distillation.DistillationModule(teacher_model_id=script_args.model, student_model_id=script_args.student_model, 
                                                     dataset=script_args.dataset, sub_dataset=script_args.dataset_subtask)
distillationObject.run_experiment()

pruningObject = pruning.PruneModel(model_id=script_args.model, dataset_id=script_args.dataset, dataset_subsetid=script_args.dataset_subtask)
pruningObject.run_experiment()

gptqQuantizationObject = GPTQQuantizer.gptqQuantization(model_id=script_args.model, dataset_id=script_args.dataset, dataset_subsetid=script_args.dataset_subtask, model_type=script_args.model_type)
gptqQuantizationObject.run_experiment()

# awqQuantizationObject = 
# awqQuantizationObject.run_experiment()

results = pd.DataFrame([
    staticQuantizationObject.results_4bit, 
    staticQuantizationObject.results_8bit, 
    distillationObject.results_distillation, 
    pruningObject.results, 
    gptqQuantizationObject.results_gptq
    # awqQuantizationObject.results
    ])

model_name = script_args.model.split("/")[-1]
results.to_csv(f"{model_name}_results.csv", index=False)
