from CompressionMethods import static_quantization, distillation, pruning, GPTQQuantizer
from dataclasses import dataclass, field
from typing import Optional
from dataclasses import field
import pandas as pd
from transformers import HfArgumentParser
import pandas as pd

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    model: str = field(default="heegyu/TinyLlama-augesc-context", metadata={"help": "The model location (huggingface or local) to use for experimentation"})
    dataset: str = field(default="heegyu/augesc", metadata={"help": "The dataset used to assess model's inference speed and accuracy."})
    model_type: str = field(default="llama", metadata={"help":"the type of model being experimented on"})
    dataset_subtask: Optional[str] = field(default=None, metadata={"help": "Path to save all the compressed model, if required"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print("Running with the following arguments:")
print(f"--model: {script_args.model}")
print(f"--dataset: {script_args.dataset}")
print(f"--dataset_subtask: {script_args.dataset_subtask}")
print(f"--model_type: {script_args.model_type}")

staticQuantizationObject = static_quantization.staticQuantization(model_id=script_args.model, dataset_id=script_args.dataset, dataset_subsetid=script_args.dataset_subtask)
staticQuantizationObject.run_experiment()

pruningObject = pruning.PruneModel(model_id=script_args.model, dataset_id=script_args.dataset, dataset_subsetid=script_args.dataset_subtask)
pruningObject.run_experiment()

gptqQuantizationObject = GPTQQuantizer.gptqQuantization(model_id=script_args.model, dataset_id=script_args.dataset, dataset_subsetid=script_args.dataset_subtask, model_type=script_args.model_type)
gptqQuantizationObject.run_experiment()

results = pd.DataFrame([
    staticQuantizationObject.results_4bit, 
    staticQuantizationObject.results_8bit, 
    # distillationObject.results, 
    pruningObject.results, 
    gptqQuantizationObject.results_gptq])

model_name = script_args.model.split("/")[-1]
results.to_csv(f"{model_name}_results.csv", index=False)
