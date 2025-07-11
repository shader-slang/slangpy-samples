import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Configuration ---
input_size = 3
hidden_size = 16
output_size = 1
lr = 1e-3
epochs = 2000
samples = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Generate Data ---
torch.manual_seed(42)
x = torch.randn(samples, input_size).to(device)
y = (x ** 2).sum(dim=1, keepdim=True).to(device)  # f(x, y, z) = x^2 + y^2 + z^2

# --- Model ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)

# --- Loss and Optimizer ---
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Training Loop ---
loss_history = []

for step in range(epochs):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if step % 200 == 0 or step == epochs - 1:
        print(f"[{step}] Loss: {loss.item():.6f}")

# --- Plotting Loss Curve ---
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

# --- Evaluation ---
with torch.no_grad():
    x_test = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    y_true = (x_test ** 2).sum(dim=1, keepdim=True)
    y_pred = model(x_test)
    print(f"\nTest input: {x_test.cpu().numpy()}")
    print(f"True output: {y_true.item():.4f}, Predicted output: {y_pred.item():.4f}")
