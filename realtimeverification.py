# Imports
import cv2
import os 
import numpy as np

import torch 
import torch.optim as optim

from model import SiameseNeuralNetwork
from loadimages import preprocess

# Specify device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path of saved model
model_path = ""

# Initialize the siamese neural network
snn = SiameseNeuralNetwork().to(device)

# Load the model
snn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['state_dict'])

# Verification function
def verify(detection_threshold, verification_threshold):
    '''
    detection_threshold: metric above which a prediction is considered positive
    verification_threshold: proportion of positive predictions / total positive samples
    '''
    results = []
    # Read the image 
    input_img = preprocess(os.path.join('application_data', 'input_images', 'input_image.jpg'))
    
    #Convert to tensor 
    input_tensor = torch.tensor(input_img).unsqueeze(0).to(device)
    
    # Loop through all the verification images
    verification_dir = os.path.join('application_data', 'verification_images')
    snn.eval()
    with torch.no_grad():
        for image in os.listdir(verification_dir):
            # Preprocess verification image
            validation_img = preprocess(os.path.join(verification_dir, image))

            # Convert to tensor and add batch dimension
            validation_tensor = torch.tensor(validation_img).unsqueeze(0).to(device)
            
             # Make prediction
            result = snn(input_tensor, validation_tensor)
            results.append(result.cpu().numpy()[0][0])

    # Detection and verification threshold
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold
    return results, verified 

# RealTime Face verification
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Resize frame 
    frame = frame[150:150+250, 190:190+250, :]

    cv2.imshow('Verification', frame)

    # Verification trigger
    if cv2.waitKey(10) & 0XFF == ord('v'):
        # Save input image
        cv2.imwrite(os.path.join('application_data', 'input_images', 'input_image.jpg'), frame)
        # Verification function
        results, verified = verify(0.4, 0.7)
        print(verified)

    # Quit if q is pressed
    if cv2.waitKey(10) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()