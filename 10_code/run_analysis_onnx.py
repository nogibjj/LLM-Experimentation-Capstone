import argparse
from NetworkExMethods import ONNXconverter

def main():
    parser = argparse.ArgumentParser(description='Convert and evaluate models to ONNX format.')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Model checkpoint for conversion.')
    parser.add_argument('--tokenizer_checkpoint', type=str, help='Tokenizer checkpoint, defaults to model checkpoint if not specified.')
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to save the ONNX model.')
    parser.add_argument('--sample_input', type=str, required=True, help='Sample input text for model conversion.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for conversion ("cpu" or "cuda").')
    parser.add_argument('--load_method', type=str, default='huggingface', help='Choose the way you want to load model. Huggingface is reconmmended')

    args = parser.parse_args()

    # Use the same tokenizer loading method as model loading for simplicity
    tokenizer_load_method = 'huggingface' if 'huggingface' in args.model_checkpoint else 'local'
    
    # Initialize the ONNX converter class
    converter = ONNXconverter(model_checkpoint=args.model_checkpoint,
                              model_load_method='huggingface',
                              device=args.device)
    
    # Perform the conversion and evaluation
    converter.convert_and_evaluate(tokenizer_load_method=tokenizer_load_method,
                                   sample_input=args.sample_input,
                                   model_checkpoint=args.model_checkpoint,
                                   tokenizer_checkpoint=args.tokenizer_checkpoint or args.model_checkpoint,
                                   onnx_path=args.onnx_path,
                                   device=args.device)

if __name__ == '__main__':
    main()
