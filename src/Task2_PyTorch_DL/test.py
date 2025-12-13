import torch

# Define some sample data
data = [1, 2, 3, 4, 5]

# Create a PyTorch tensor from the data
tensor_data = torch.tensor(data)

# Print the tensor
print(tensor_data)

# Get some information about the tensor
print(f"Tensor shape: {tensor_data.shape}")
print(f"Tensor data type: {tensor_data.dtype}")
