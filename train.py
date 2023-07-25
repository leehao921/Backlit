import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from unet import UNet
from data_set import Dataset

# Define the loss function

# def loss_function(output, target, input, ground_truth, lambd):
#     perceptual_loss = F.mse_loss(output, target)

#     batch_loss = torch.mean((input - output)**2 * torch.exp(lambd * torch.abs(input - ground_truth)))

#     total_loss = batch_loss + perceptual_loss
#     return total_loss

def dice_loss(y_pred, y_true, smooth=1e-5):
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)

    intersection = torch.sum(y_pred_flat * y_true_flat)
    union = torch.sum(y_pred_flat) + torch.sum(y_true_flat)

    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score

    return dice_loss

def train(model, train_loader, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        outputs = model(inputs)
        loss = dice_loss(outputs, targets)

        loss.backward()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

if __name__ == "__main__":  
    # Define hyperparameters
    num_epochs = 1
    batch_size = 32
    initial_lr = 0.001

    # Create the U-Net model
    #!  the device for azure 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet(3,1)
    model = unet.to(device)

    # Create the optimizer and loss function
    #? The entire network is optimized by Adam optimizer. The initial learning rate is set to 0.001 for the first 100 epochs and decreases by half every 10 epochs.


  

    #!  the data_path for azure 
    data_path = "/Users/felix/Documents/Internship實習/2023工研院/Original/train/"
    train_loader = DataLoader(Dataset(data_path), batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader , device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    

    ################################################################
