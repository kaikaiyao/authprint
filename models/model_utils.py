import copy
import os
import pickle


def save_finetuned_model(
    model, 
    path, 
    filename
):
    model_cpu = copy.deepcopy(model).cpu()
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(model_cpu, f)


def load_finetuned_model(
    path
):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def clone_model(model):
    """
    Clones a model and ensures all parameters in the cloned model require gradients.
    
    Args:
        model (torch.nn.Module): The model to clone.
    
    Returns:
        torch.nn.Module: The cloned model with all parameters requiring gradients.
    """
    # Use deepcopy to create a copy of the model
    cloned_model = copy.deepcopy(model)
    
    # Ensure the cloned model is in training mode
    cloned_model.train()
    
    # Ensure all parameters in the cloned model require gradients
    for param in cloned_model.parameters():
        param.requires_grad = True
    
    return cloned_model