"""
PyTorch and TorchVision Installation Test
Tests whether torch and torchvision are properly installed and functional
"""

import sys


def test_torch_import():
    """Test if torch can be imported"""
    print("=" * 60)
    print("Testing torch import...")
    print("=" * 60)
    try:
        import torch
        print(f"✓ torch imported successfully")
        print(f"  Version: {torch.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import torch: {e}")
        return False


def test_torchvision_import():
    """Test if torchvision can be imported"""
    print("\n" + "=" * 60)
    print("Testing torchvision import...")
    print("=" * 60)
    try:
        import torchvision
        print(f"✓ torchvision imported successfully")
        print(f"  Version: {torchvision.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import torchvision: {e}")
        return False


def test_torch_basics():
    """Test basic torch tensor operations"""
    print("\n" + "=" * 60)
    print("Testing torch basic operations...")
    print("=" * 60)
    try:
        import torch
        
        # Test tensor creation
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✓ Created tensor: {x}")
        
        # Test tensor operations
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        print(f"✓ Tensor addition: {x} + {y} = {z}")
        
        # Test matrix multiplication
        A = torch.randn(3, 3)
        B = torch.randn(3, 3)
        C = torch.matmul(A, B)
        print(f"✓ Matrix multiplication: shape {A.shape} × {B.shape} = {C.shape}")
        
        # Test gradients
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2
        y.backward()
        print(f"✓ Gradient computation: dy/dx at x=2 is {x.grad.item()}")
        
        return True
    except Exception as e:
        print(f"✗ torch basic operations failed: {e}")
        return False


def test_cuda_availability():
    """Test if CUDA is available"""
    print("\n" + "=" * 60)
    print("Testing CUDA availability...")
    print("=" * 60)
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            print(f"  Device count: {torch.cuda.device_count()}")
            
            # Test tensor on GPU
            x = torch.randn(3, 3).cuda()
            print(f"✓ Successfully created tensor on GPU")
            return True
        else:
            print("✓ CUDA is not available (CPU mode)")
            return True
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False


def test_torchvision_transforms():
    """Test torchvision transforms"""
    print("\n" + "=" * 60)
    print("Testing torchvision transforms...")
    print("=" * 60)
    try:
        from torchvision import transforms
        from PIL import Image
        import torch
        import numpy as np
        
        # Create a sample image
        sample_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        sample_image = Image.fromarray(sample_array)
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Apply transforms
        transformed = transform(sample_image)
        print(f"✓ Image transforms applied successfully")
        print(f"  Input shape: {sample_image.size}")
        print(f"  Output shape: {transformed.shape}")
        
        return True
    except Exception as e:
        print(f"✗ torchvision transforms failed: {e}")
        return False


def test_torchvision_models():
    """Test torchvision models"""
    print("\n" + "=" * 60)
    print("Testing torchvision models...")
    print("=" * 60)
    try:
        from torchvision import models
        import torch
        
        # Load a pretrained model
        model = models.resnet18(pretrained=False)
        print(f"✓ ResNet18 model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ torchvision models failed: {e}")
        return False


def test_nn_module():
    """Test torch.nn module"""
    print("\n" + "=" * 60)
    print("Testing torch.nn neural network module...")
    print("=" * 60)
    try:
        import torch
        import torch.nn as nn
        
        # Define a simple neural network
        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 2)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # Create and test the network
        model = SimpleNet()
        x = torch.randn(4, 10)
        output = model(x)
        
        print(f"✓ Neural network created and executed successfully")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ torch.nn module test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("# PyTorch and TorchVision Installation Tests")
    print("#" * 60)
    
    results = []
    
    results.append(("torch import", test_torch_import()))
    results.append(("torchvision import", test_torchvision_import()))
    
    if results[0][1]:  # Only run further tests if torch imported
        results.append(("torch basics", test_torch_basics()))
        results.append(("CUDA availability", test_cuda_availability()))
        results.append(("torch.nn module", test_nn_module()))
    
    if results[1][1]:  # Only run torchvision tests if imported
        results.append(("torchvision transforms", test_torchvision_transforms()))
        results.append(("torchvision models", test_torchvision_models()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "#" * 60)
    if all_passed:
        print("# All tests PASSED! PyTorch and TorchVision are working correctly.")
    else:
        print("# Some tests FAILED. Please check the errors above.")
    print("#" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())