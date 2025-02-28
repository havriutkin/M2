import argparse
import torch 

def parse_args():
    parser = argparse.ArgumentParser(description="Run TorchScript Model")
    parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the TorchScript saved model file (e.g., scripted_model.pt)")
    parser.add_argument("--input", type=str, required=True,
                    help="Comma-separated input values for the input layer")
    return parser.parse_args()

args = parse_args()
model_path = args.model_path
input_str = args.input

input_values = list(map(float, input_str.split(',')))
input_tensor = torch.tensor(input_values).unsqueeze(0)

model = torch.jit.load(model_path, 
                    map_location=torch.device('cpu'))
model.eval()

output = model(input_tensor)
output = output.tolist()
print(output)
