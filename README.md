# One-Shot-Image-Recognition
In this project one-shot-image-recognition is performed with the help of the Siamese Neural Network(SNN). The aim of the project is to carry out face recognition with the help of SNN in real time. SNN use two streams/inputs - the anchor image and the image in consideration(positive and negative). Both the pictures are passed through the consequent layers and the L1 distance(in this project) between the anchor image and the image being considered is calculated. The model is then used to predict if the face is **verified or not**.

# Datset
The dataset for this project involves having an anchor, positive and negative dataset. For the anchor and positive datasets the webcam is used to capture pictures of the face we want to store. For the negative dataset, the labeled faces in the wild(lfw) is used.
Link to the lfw dataset: https://vis-www.cs.umass.edu/lfw/

# Model 
PyTorch is used to implement the Siamese Neural Network. The model has about 77M parameters.

![image](https://github.com/user-attachments/assets/669301e1-3d3b-4d05-bc27-9fb19f865ddf)

For more reference, the architecture used in this project is shown in the below image:
![image](https://github.com/user-attachments/assets/4dba1dcf-ad1d-4213-b272-15780fd3a368)

# Results 
The model achieved an accuracy of 95.83% on the test data curated. 

# Real Time Verification 
![image](https://github.com/user-attachments/assets/ca251ae7-2469-4b72-b0c1-0ebe7e8e7d31)
