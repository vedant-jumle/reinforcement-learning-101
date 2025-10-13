# Trained Models

This directory stores trained policy models for the drone landing task.

## Files

- `*.pth` - PyTorch state dictionaries (recommended for loading)
- `*.pt` - Complete model files (includes architecture)

## Usage

### Saving a Model

```python
import torch

# Save state dict (recommended)
torch.save(policy.state_dict(), 'models/my_policy.pth')

# Or save complete model
torch.save(policy, 'models/my_policy.pt')
```

### Loading a Model

```python
import torch
from your_policy_module import DroneGamerBoi

# Load state dict
policy = DroneGamerBoi()
policy.load_state_dict(torch.load('models/my_policy.pth'))
policy.eval()

# Or load complete model
policy = torch.load('models/my_policy.pt')
policy.eval()
```

## Note

Model files are ignored by git (see `.gitignore`). This keeps the repository lightweight and prevents accidentally committing large binary files.

Share trained models separately via cloud storage or model registries.
