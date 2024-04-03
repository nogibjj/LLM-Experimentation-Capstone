from CompressionMethods import static_quantization, distillation, GPTQQuantizer, pruning
from dataclasses import dataclass, field
from typing import Optional
from dataclasses import field
import pandas as pd
# Libary Imports
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    model: str = field(default="adjohn1313/bert-base-finetuned", metadata={"help": "The model location (huggingface or local) to use for experimentation"})
    dataset: str = field(default="sem_eval_2018_task_1", metadata={"help": "The dataset used to assess model's inference speed and accuracy."})
    dataset_subtask: Optional[str] = field(default="subtask5.english", metadata={"help": "Path to save all the compressed model, if required"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print("Running with the following arguments:")
print(f"--model: {script_args.model}")
print(f"--dataset: {script_args.dataset}")
print(f"--dataset_subtask: {script_args.dataset_subtask}")

staticQuantizationObject = static_quantization.staticQuantization(model_id=script_args.model, dataset_id=script_args.dataset, dataset_subsetid=script_args.dataset_subtask)
staticQuantizationObject.run_experiment()

distillationObject = distillation.KnowledgeDistillationTrainer(model_id=script_args.model, dataset_id=script_args.dataset, dataset_subsetid=script_args.dataset_subtask)
distillationObject.run_experiment()

pruningObject = pruning.PruneModel(model_id=script_args.model, dataset_id=script_args.dataset, dataset_subsetid=script_args.dataset_subtask)
pruningObject.run_experiment()

gptqQuantizationObject = GPTQQuantizer.GPTQQuantizer(model_id=script_args.model, dataset_id=script_args.dataset, dataset_subsetid=script_args.dataset_subtask)
gptqQuantizationObject.run_experiment()

results = pd.DataFrame([
    staticQuantizationObject.results_4bit, 
    staticQuantizationObject.results_8bit, 
    distillationObject.results, 
    pruningObject.results, 
    gptqQuantizationObject.results])

results.to_csv("bert_results.csv", index=False)