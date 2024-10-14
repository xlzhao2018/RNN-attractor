import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the LSTM model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        out = torch.tanh(out)  # Add activation function
        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.randn(1, batch_size, self.hidden_size),
                torch.randn(1, batch_size, self.hidden_size))

# Generate input sequence (e.g., sine wave)
def generate_sequence(seq_len, batch_size):
    t = np.linspace(0, np.pi * 4, seq_len)
    sequence = np.sin(t)
    sequence = sequence.reshape(-1, 1)
    sequence = np.tile(sequence, (batch_size, 1, 1))  # Batch replication
    return torch.Tensor(sequence)

# Hyperparameters
input_size = 1
hidden_size = 50  # Increased hidden units
output_size = 1
seq_len = 20
batch_size = 16
num_epochs = 1000
learning_rate = 0.001  # Increased learning rate

# Initialize the model, loss function, and optimizer
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Store losses for plotting later
losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Generate sequences
    input_seq = generate_sequence(seq_len, batch_size)
    target_seq = input_seq.clone()  # Predict same sequence
    
    # Initialize hidden state
    hidden = model.init_hidden(batch_size)
    
    # Forward pass
    output_seq, hidden = model(input_seq, hidden)
    
    # Compute loss
    loss = criterion(output_seq, target_seq)
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Store loss for plotting
    losses.append(loss.item())
    
    # Print loss periodically
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, evaluate the model
model.eval()
test_seq = generate_sequence(seq_len, 1)  # Single test sequence

# Initialize hidden state for evaluation
hidden = model.init_hidden(1)
hidden = (hidden[0].detach(), hidden[1].detach())  # Detach hidden state

# Predict the sequence
with torch.no_grad():
    pred_seq, hidden = model(test_seq, hidden)

# Predict beyond input sequence
future_steps = seq_len  # Number of steps to predict beyond the input sequence
current_input = test_seq[:, -1:, :]  # Start with the last input
pred_persistent = []

with torch.no_grad():
    for _ in range(future_steps):
        pred, hidden = model(current_input, hidden)
        pred_persistent.append(pred.item())
        current_input = pred.unsqueeze(1)

# Convert to NumPy for plotting
pred_seq = pred_seq.squeeze().numpy()
test_seq = test_seq.squeeze().numpy()
pred_persistent = np.array(pred_persistent)

# Plot input and predictions
plt.figure(figsize=(10, 6))
plt.plot(test_seq, label='Input Sequence', color='blue')
plt.plot(pred_seq, label='Predicted Sequence', color='green')
plt.plot(np.arange(seq_len, seq_len + future_steps), pred_persistent, label='Persistent Activity', color='red')
plt.title('Attractor RNN: Simplified LSTM Predictions')
plt.legend()
plt.show()

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(losses, label="Training Loss")
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

