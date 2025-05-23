import torch
from torch_mlir import torchscript
import os
import argparse

os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers/cache/'

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to MLIR.")
    parser.add_argument("out_mlir_path", nargs="?", default="./output/01_tosa.mlir", help="Path to write the MLIR file to.")
    dialect_choices = ["tosa", "linalg-on-tensors", "torch", "raw", "mhlo"]
    parser.add_argument("--dialect", default="linalg-on-tensors", choices=dialect_choices, help="Dialect to use for lowering.")
    
    args = parser.parse_args()
    return args

class ToyCNN(torch.nn.Module):
    def __init__(self):
        super(ToyCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(1 * 4 * 4, 4)  # Assumes input image size of 4x4

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc(x)
        return x

def main():
    args = parse_args()

    # Create a model
    model = ToyCNN()
    print(model)

    # Prepare directory and input data
    os.makedirs(os.path.dirname(args.out_mlir_path), exist_ok=True)
    in_shape = {'bs': 4, 'c': 1, 'h': 4, 'w': 4}
    input_data = torch.randn(in_shape['bs'], in_shape['c'], in_shape['h'], in_shape['w'])

    # Generate the MLIR module
    module = torchscript.compile(model, input_data, output_type=args.dialect, use_tracing=True)
    with open(args.out_mlir_path, "w", encoding="utf-8") as outf:
        outf.write(str(module))
    
if __name__ == "__main__":
    main()
