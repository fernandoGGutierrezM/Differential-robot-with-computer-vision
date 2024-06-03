import torch
import torch.nn as nn

# Define a dummy model architecture
# This should match the architecture of your saved model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)  # Example layer

    def forward(self, x):
        return self.linear(x)

def load_model(model_path):
    # Ensure the model architecture is defined
    model = SimpleModel()
    # Load the saved model parameters
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def make_inference(model, input_data):
    # Convert input data to a PyTorch tensor
    input_tensor = torch.FloatTensor(input_data)
    # Make an inference
    with torch.no_grad():
        output = model(input_tensor)
    return output

# Path to the .pt model file
model_path = 'model.pt'

# Load the model
model = load_model(model_path)

# Dummy input data (adjust size according to your model input)
dummy_input = [0.5] * 10  # Example input of size 10

# Perform inference
output = make_inference(model, dummy_input)
print("Model output:", output)
