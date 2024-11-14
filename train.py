# Imports 
import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
import time 
from sklearn.metrics import precision_score, recall_score

import torch 
from torch import unsqueeze
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset, random_split

from negativeimages import ANC_PATH, POS_PATH, NEG_PATH
from loadimages import train_dataloader, test_dataloader
from model import SiameseNeuralNetwork

# Set seed 
torch.manual_seed(42)

# Use device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Checkpoint function
def save_checkpoint(state, checkpoint_dir):
    checkpoint = os.path.join(checkpoint_dir, 'ckpt.pt')
    torch.save(state, checkpoint)



# Model 
model = SiameseNeuralNetwork().to(device)

# Loss function
loss_type = nn.BCELoss()

# Optimizer 
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define checkpoint dir path
checkpoint_dir = "E:\\code\\projects\\OneShotImageRecognition\\training_checkpoints"


# Training 
epochs = 40

# Train step function
def train_batch(batch):
    # get images and labels
    images, labels = batch[:2], batch[2]
    
    # Set inputs to cuda
    img1 = images[0].to(device)
    img2 = images[1].to(device)
    
    # Predict 
    pred = model(img1, img2)
    loss = loss_type(pred, labels.unsqueeze(1).to(device))

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# Train function
def train_model(train_loader, epochs=epochs, model=model):
    model.train()
    start_time = time.time()
    training_losses = []
    for epoch in range(epochs):
        # Set progress-bar
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

        for batch_idx, batch in loop:
            # Calculate loss
            loss = train_batch(batch)

            # Update progress bar
            loop.set_description(f" Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss = loss.item())
        
        # Keep a track of losses every epoch
        training_losses.append(loss.item())

        # Save checkpoint 
        if epoch%20==0:    
            checkpoint = {'state_dict':model.state_dict, 'optimizer': optimizer.state_dict}
            save_checkpoint(checkpoint, checkpoint_dir)
    
    # Release cache
    torch.cuda.empty_cache()

    print("Finished training. Training took {0:.2f} minutes.".format((time.time()-start_time)/60))
    return training_losses
        
    
training_losses = train_model(train_dataloader)

# Training loss curve
plt.plot(range(epochs), training_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# Testing 
def test_model(test_loader, model=model, epochs=epochs):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        loop = tqdm(enumerate(test_dataloader), total = len(test_dataloader), leave = False)
        for batchidx, batch in loop:
            images, labels = batch[:2], batch[2]
            
            # Convert the inputs and labels to device
            img1 = images[0].to(device)
            img2 = images[1].to(device)
            labels = labels.to(device)

            # Predictions
            pred = model(img1, img2)

            # Convert the prediction to binary
            pred_binary = (pred.squeeze() >= 0.5).float()
            total += labels.size(0)
            correct += (pred_binary == labels).sum().item()

    # Calculate accuracy
    accuracy = (correct/total)*100
    return accuracy

accuracy = test_model(test_dataloader)
print("Accuracy is {0:.3f}%".format(accuracy))

# Save the model 
# final_model = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
# torch.save(final_model, 'siamesemodel.pt')
# print("Succesfully saved the model.")