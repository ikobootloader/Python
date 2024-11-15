import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from time import time

def generate_complex_data(n_samples, input_size):
    x = torch.randn(n_samples, input_size)
    y1 = torch.sin(2 * x[:, 0]) * torch.cos(3 * x[:, 1]) + torch.tanh(x[:, 2] * x[:, 3])
    y2 = torch.exp(-torch.abs(x[:, 0])) * torch.sin(5 * x[:, 1]) + torch.sigmoid(x[:, 2] + x[:, 3])
    y3 = torch.log(torch.abs(x[:, 0]) + 1) * torch.cos(4 * x[:, 1]) + torch.relu(x[:, 2] - x[:, 3])
    y = torch.stack([y1, y2, y3], dim=1)
    return x, y

class NoiseRobustAttractorLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_attractors=4):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.head_dim = hidden_size // 4
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, self.head_dim * 2),
                nn.LayerNorm(self.head_dim * 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.head_dim * 2, self.head_dim)
            ) for _ in range(4)
        ])
        
        self.attractors = nn.ParameterList([
            nn.Parameter(torch.randn(num_attractors * 2, self.head_dim) / np.sqrt(self.head_dim))
            for _ in range(4)
        ])
        
        self.dynamics = nn.ParameterList([
            nn.Parameter(torch.eye(self.head_dim) + 0.05 * torch.randn(self.head_dim, self.head_dim))
            for _ in range(4)
        ])
        
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, 1),
                nn.Sigmoid()
            ) for _ in range(4)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, input_size)
        )
    
    def compute_head_output(self, x, head_idx):
        state = self.heads[head_idx](x)
        dists = torch.cdist(state, self.attractors[head_idx])
        attention_weights = F.softmax(-dists / np.sqrt(self.head_dim), dim=-1)
        attractor_influence = torch.matmul(attention_weights, self.attractors[head_idx])
        dynamics = torch.matmul(state, self.dynamics[head_idx])
        gate = self.gates[head_idx](x)
        return state + gate * 0.1 * (attractor_influence - state + torch.tanh(dynamics))
    
    def forward(self, x):
        x = self.norm(x)
        head_outputs = [self.compute_head_output(x, i) for i in range(4)]
        combined = torch.cat(head_outputs, dim=-1)
        return x + self.output(combined)

class NoiseRobustAttractorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.layers = nn.ModuleList([
            NoiseRobustAttractorLayer(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, output_size)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=4, num_layers=3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size*2, output_size)
        )
        
    def forward(self, x):
        x = self.input_projection(x).unsqueeze(1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        return self.output_layers(x.squeeze(1))

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def train_model(model, x, y, epochs=100, batch_size=128, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, total_epochs=epochs)
    
    n_samples = len(x)
    losses = []
    
    start_time = time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        permutation = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x[indices], y[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = F.mse_loss(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
        scheduler.step(epoch)
        epoch_loss = running_loss / (n_samples / batch_size)
        losses.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    
    training_time = time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    return losses, training_time

def evaluate_model(model, x, y, device='cpu'):
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        mse = F.mse_loss(predictions, y).item()
    return mse

def compare_models(x_train, y_train, device='cpu'):
    input_size = x_train.shape[1]
    hidden_size = 64
    output_size = y_train.shape[1]
    
    noise_levels = [0.0, 0.1, 0.2]
    results = {
        'transformer': {},
        'attractor': {}
    }
    
    for noise in noise_levels:
        print(f"\nTesting with noise level: {noise}")
        noisy_x = x_train + torch.randn_like(x_train) * noise
        
        print("Training transformer...")
        transformer = TransformerModel(input_size, hidden_size, output_size).to(device)
        transformer_losses, t_time = train_model(transformer, noisy_x, y_train, device=device)
        transformer_mse = evaluate_model(transformer, noisy_x, y_train, device)
        
        print("Training attractor network...")
        attractor = NoiseRobustAttractorNetwork(input_size, hidden_size, output_size).to(device)
        attractor_losses, a_time = train_model(attractor, noisy_x, y_train, device=device)
        attractor_mse = evaluate_model(attractor, noisy_x, y_train, device)
        
        print(f"Transformer - MSE: {transformer_mse:.4f}, Time: {t_time:.2f}s")
        print(f"Attractor - MSE: {attractor_mse:.4f}, Time: {a_time:.2f}s")
        
        results['transformer'][noise] = (transformer_losses, transformer_mse, t_time)
        results['attractor'][noise] = (attractor_losses, attractor_mse, a_time)
    
    return results

def plot_results(results):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    for noise in results['transformer'].keys():
        t_losses = results['transformer'][noise][0]
        a_losses = results['attractor'][noise][0]
        plt.plot(t_losses, label=f'Transformer (noise={noise})', linestyle='--')
        plt.plot(a_losses, label=f'Attractor (noise={noise})')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Learning Curves')
    
    plt.subplot(122)
    noise_levels = list(results['transformer'].keys())
    t_mse = [results['transformer'][n][1] for n in noise_levels]
    a_mse = [results['attractor'][n][1] for n in noise_levels]
    
    x = np.arange(len(noise_levels))
    width = 0.35
    
    plt.bar(x - width/2, t_mse, width, label='Transformer')
    plt.bar(x + width/2, a_mse, width, label='Attractor')
    plt.xticks(x, [f'Noise {n}' for n in noise_levels])
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Final MSE Comparison')
    
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Generating data...")
    n_samples = 1000
    input_size = 5
    x_train, y_train = generate_complex_data(n_samples, input_size)
    
    x_train = (x_train - x_train.mean(dim=0)) / (x_train.std(dim=0) + 1e-7)
    y_train = (y_train - y_train.mean(dim=0)) / (y_train.std(dim=0) + 1e-7)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    results = compare_models(x_train, y_train, device)
    plot_results(results)

if __name__ == "__main__":
    main()