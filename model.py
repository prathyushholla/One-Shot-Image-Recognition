import torch 
import torch.nn as nn 
from torchsummary import summary 
# Create the embedding network
class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        # Define a dropout layer
        self.dropout = nn.Dropout(0.2)

        # First block 
        self.conv1 = nn.Conv2d(3, 64, 10) # input = 3, output = 64, kerna = 10x10
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # kernal = 2x2
        

        # Second block
        self.conv2 = nn.Conv2d(64, 128, 7) # input = 64, output = 128, kernal = 7x7
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # kernal = 2x2
        self.bn2 = nn.BatchNorm2d(128)

        # Third block
        self.conv3 = nn.Conv2d(128, 128, 4) # input = 128, output = 128, kernal = 4x4
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2) # kernal size = 2x2

        # Fourth block
        self.conv4 = nn.Conv2d(128, 256, 4) # input = 128, output = 256, kernal = 4x4
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(256)
        # Flatten 
        self.flatten = nn.Flatten() # Flatten the last layer 

        # Dense layer 
        self.dense = nn.Linear(9216, 4096)
        # self.fc1 = nn.Linear(5048, 4096)

        # Output Layer with sigmoid 
        self.output = nn.Sigmoid()

    def forward(self, x):
        # Input image = 105x105

        # First block
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        

        # Second block 
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        # Third block 
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Fourth block
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        # Flatten 
        x = self.flatten(x)

        # Dense layer
        x = self.dense(x)
        # x = self.fc1(x)

        return self.output(x)
    
# Create the Siamese Neural Network 

class SiameseNeuralNetwork(nn.Module):
    def __init__(self):
        super(SiameseNeuralNetwork, self).__init__()
        self.embedding = EmbeddingNetwork()
        self.classifier = nn.Linear(4096,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img1, img2):
        # Get embeddings for both images 
        embedding1 = self.embedding(img1)
        embedding2 = self.embedding(img2)
        
        # Calculate the L1 distance between the two embeddings
        distance = torch.abs(torch.sub(embedding1, embedding2))

        return self.sigmoid(self.classifier(distance))
    
if __name__ == '__main__':
    # Instantiate the model
    model = SiameseNeuralNetwork()
    
    # Summarize the model
    summary(model.cuda(), [(3, 105, 105),(3, 105, 105)])

