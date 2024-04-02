# Why ONNX

## More Universal Deployment

ONNX (Open Neural Network Exchange) enables seamless model deployment across diverse AI frameworks and hardware, ensuring interoperability and optimization. By adopting ONNX, developers can transfer models effortlessly between platforms like TensorFlow and PyTorch to various devices, from cloud services to edge computing. This compatibility enhances performance, leveraging specific hardware accelerations like GPUs and TPUs. As a result, ONNX streamlines the path from model development to production, maintaining consistent performance and broadening deployment possibilities.

## Inference Time Optimization

Based on our experiment findings, five models, including three with transformer architectures and two BERT models, achieved an average inference speed optimization of 2X.

# ONNX Runtime GPU Support

To leverage GPU acceleration with ONNX Runtime, ensure you install the correct version of ONNX Runtime that corresponds with your CUDA version. Using the wrong versions may result in the ONNX model defaulting to CPU execution, even if you specify the GPU (`cuda`) as the target device.

## Installation Instructions

**For GPU Usage:** Replace `onnxruntime` with `onnxruntime-gpu` for GPU support. Install via pip using:

```sh
pip install onnxruntime-gpu

**CUDA Version Compatibility**
Personal Machines (Common CUDA Version 11.X): The standard onnxruntime-gpu package supports CUDA 11.X, which is widely used on personal devices. It can be installed directly using pip as shown above.

Virtual Machines and Cloud Platforms (CUDA Version 12.X): For platforms commonly using CUDA 12.X, such as Codespace and Google Colab, a specific version of ONNX Runtime needs to be installed. Use the following command to install:

pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

Additional Resources
For more detailed information about ONNX Runtime and its compatibility with CUDA, please visit the ONNX Runtime documentation.

