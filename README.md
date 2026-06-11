# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
A neural network regression model is used to predict continuous numerical values based on input data. Unlike classification problems that assign inputs to categories, regression focuses on estimating real-valued outputs such as price, temperature, or demand.

The problem statement in this context involves developing a model that can learn the relationship between input features (independent variables) and a continuous target variable (dependent variable) using a neural network. The dataset typically contains multiple input attributes, and the goal is to train the model so that it can accurately predict the output for unseen data.

A neural network consists of layers of interconnected neurons, including an input layer, one or more hidden layers, and an output layer. Each neuron processes inputs using weights and biases, applies an activation function, and passes the result to the next layer. During training, the network adjusts these weights using optimization techniques like gradient descent to minimize the error between predicted and actual values.

In regression tasks, commonly used loss functions include Mean Squared Error (MSE) or Mean Absolute Error (MAE), which measure how far the predictions are from the true values. The model learns by iteratively updating its parameters to reduce this loss.

The main objective of this problem is to:

Build a neural network model

Train it using the dataset
Evaluate its performance

Use it to make accurate continuous predictions

This approach is widely used in applications such as house price prediction, stock forecasting, and demand estimation, where outputs are numerical rather than categorical.


## Neural Network Model
<img width="1116" height="757" alt="image" src="https://github.com/user-attachments/assets/bf1e1aad-fecb-4ecb-a3cf-1e1f04db0f82" />


## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2: 
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3: 
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4: 
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.

## PROGRAM

### Name: Bhuvaneshwaran H

### Register Number: 212223240018

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 =nn.Linear(input_size,32)
        self.fc2 =nn.Linear(32, 16)
        self.fc3 =nn.Linear(16, 8)
        self.fc4 =nn.Linear(8, 4)
    def forward(self, x):
      #Include your code here
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x    
def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
train_model(model, train_loader, criterion, optimizer, epochs=100)
```

### Dataset Information
<img width="1010" height="217" alt="image" src="https://github.com/user-attachments/assets/b715c9ba-ea06-4a29-b9b9-0fa3e86b398a" />


### OUTPUT

## Confusion Matrix

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/6d0a5c69-ffab-4fa9-aef9-c49bb85acba1" />



## Classification Report
<img width="473" height="311" alt="image" src="https://github.com/user-attachments/assets/f2405061-d889-454d-af13-547bd1eadebc" />


### New Sample Data Prediction
<img width="318" height="89" alt="image" src="https://github.com/user-attachments/assets/fa80f12d-231d-4162-ad64-5d69ac8d1ae2" />


## RESULT
A neural network classification model was successfully developed and tested on the given dataset with satisfactory classification performance.
