# Step 6: Adversarial Training
# Add adversarial examples for robustness and document the process

import torch
import torch.nn.functional as F

def fgsm_attack(model, loss_fn, data, target, epsilon):
    data.requires_grad = True
    output = model(*data)
    loss = loss_fn(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return perturbed_data.detach()

# Alignment Check
print('Alignment Check:')
print('- Adversarial training increases robustness against sophisticated fraud attempts.')
print('- Next: Model training, evaluation, and real-time pipeline.')
