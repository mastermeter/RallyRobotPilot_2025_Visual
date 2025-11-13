# rallyrobopilot/ml/model.py
import torch.nn as nn

class RobopilotMLP(nn.Module):
    """A flexible Multi-Layer Perceptron for the Rally Robopilot."""
    def __init__(self, input_size=15, output_size=4, hidden_layers=[128, 64], dropout_rate=0.1):
        """
        Initializes the MLP model.

        Args:
            input_size (int): The number of input features.
            output_size (int): The number of output values.
            hidden_layers (list of int): A list where each integer is the number of neurons
                                         in a hidden layer.
            dropout_rate (float): The dropout rate to apply after each hidden layer.
        """
        super(RobopilotMLP, self).__init__()
        
        # Create a list to hold all the layers
        layers = []
        # The input feature size for the first layer is the overall input_size
        in_features = input_size

        # Loop through the provided hidden layer sizes and build the network
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            # The input for the next layer is the output of the current layer
            in_features = hidden_size

        # Add the final output layer
        layers.append(nn.Linear(in_features, output_size))
        layers.append(nn.Sigmoid())

        # Create the sequential model from the list of layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)