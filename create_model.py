import torch
import torch.nn as nn

class SimpleSumModel(nn.Module):
    def forward(self, x):
        # x is expected to have shape [batch_size, num_features]
        # This returns the sum of the features for each input sample.
        return x.sum(dim=1, keepdim=True)

# Instantiate the model and set to evaluation mode.
model = SimpleSumModel()
model.eval()

# Convert the model to TorchScript.
scripted_model = torch.jit.script(model)

# Save the TorchScript model.
scripted_model.save("simple_sum_model.pt")
print("Saved TorchScript model as 'simple_sum_model.pt'")
