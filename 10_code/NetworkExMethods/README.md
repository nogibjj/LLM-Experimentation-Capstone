# Why ONNX

## More Robust Deployment

ONNX (Open Neural Network Exchange) enables seamless model deployment across diverse AI frameworks and hardware, ensuring interoperability and optimization. By adopting ONNX, developers can transfer models effortlessly between platforms like TensorFlow and PyTorch to various devices, from cloud services to edge computing. This compatibility enhances performance, leveraging specific hardware accelerations like GPUs and TPUs. As a result, ONNX streamlines the path from model development to production, maintaining consistent performance and broadening deployment possibilities.

## Inference Time Optimization

Based on our experiment findings, five models, including three with transformer architectures and two BERT models, achieved an average inference speed optimization of 2X.

# Dependency & Hardware Support

To leverage GPU acceleration with ONNX Runtime, ensure you install the correct version of ONNX Runtime that corresponds with your CUDA version. Using the wrong versions may result in the ONNX model defaulting to CPU execution, even if you specify the GPU (`cuda`) as the target device.

Install as:

```sh
!pip install onnx optimum onnxruntime
```

Import as:
```sh
import onnx
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
```


## Installation Instructions

**For GPU Usage:** Replace `onnxruntime` with `onnxruntime-gpu` for GPU support. Install via pip using:

```sh
pip install onnxruntime-gpu
```

## CUDA Version Compatibility
Personal Machines (Common CUDA Version 11.X): The standard onnxruntime-gpu package supports CUDA 11.X, which is widely used on personal devices. It can be installed directly using pip as shown above.

Virtual Machines and Cloud Platforms (CUDA Version 12.X): For platforms commonly using CUDA 12.X, such as Codespace and Google Colab, a specific version of ONNX Runtime needs to be installed. Use the following command to install:

```sh
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

# How to Use

## Convert Model
The model to convert can from Local, Global Environment, or Huggingface

Hugging face Example

```sh
converter = ONNXconverter( model_load_method = "huggingface",model_checkpoint = "elisachen/gptq-tinyllama-classification", device="cuda",op_version = 14, architecture="transformer")
results = model_to_onnx_and_evaluate(
                            tokenizer_load_method = "huggingface",
                            sample_input = x[0],
                            tokenizer_checkpoint = "elisachen/gptq-tinyllama-classification",
                            onnx_path = "model.onnx",
                            x=x,
                            y=y_true
                            )
print("Evaluation Results:", results)
```


## Additional Resources

For more detailed information about ONNX Runtime and its compatibility with CUDA, please visit the [ONNX Runtime documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

