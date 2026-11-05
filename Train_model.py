import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class FinSignDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.labels = []
        
        files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        raw_labels = []
        
        for file in files:
            # 1. Strip the .npy extension
            base = os.path.splitext(file)[0].strip()
            
            # 2. Strip out the variation suffix so _var perfectly groups with the baseline
            base = base.split('_var')[0].strip()
            
            # 3. Strip trailing clip numbers, underscores, and dots (fixes the 100K L.E. bug!)
            base = re.sub(r'[_\s]+\d{1,2}$', '', base)
            base = base.rstrip('_.').strip()
            
            if base.upper() == 'IDLE':
                base = 'Idle'
                
            raw_labels.append(base)
            tensor = np.load(os.path.join(data_path, file))
            self.data.append(tensor)
            
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(raw_labels)
        self.classes = self.label_encoder.classes_
        print(f"Loaded {len(self.data)} samples across {len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y



class FinSignTransformer(nn.Module):
    def __init__(self, input_size=222, num_classes=4, d_model=128, nhead=4, num_layers=3):
        super(FinSignTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        self.positional_encoding = nn.Parameter(torch.randn(1, 30, d_model))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256, 
            dropout=0.2, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.positional_encoding 
        
        transformer_out = self.transformer_encoder(x)
        
        pooled_out = transformer_out.mean(dim=1)
        
        out = self.fc(pooled_out)
        return out

def train_model():
    data_path = r"C:\Users\PC\Desktop\FinSign's Dataset\Extracted Tensors"
    dataset = FinSignDataset(data_path)
    
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    

    input_size = 222
    hidden_size = 128
    num_layers = 2
    num_classes = len(dataset.classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FinSignTransformer(input_size=222, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 80
    print(f"Starting training on {device} for {num_classes} financial terms...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 1. Random temporal shift (-4 to +4 frames) so sliding window alignment doesn't matter
            shift = np.random.randint(-4, 5)
            batch_x_aug = torch.roll(batch_x, shifts=shift, dims=1)
            
            # 2. Random scale multiplier (0.85x to 1.15x) + Gaussian noise
            scale_jitter = (torch.rand(batch_x.size(0), 1, 1, device=device) * 0.3) + 0.85
            noise = torch.randn_like(batch_x_aug) * 0.015
            batch_x_aug = (batch_x_aug * scale_jitter) + noise
            
            optimizer.zero_grad()
            outputs = model(batch_x_aug)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_samples += batch_y.size(0)
                correct_predictions += (predicted == batch_y).sum().item()
                
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = (correct_predictions / total_samples) * 100
            
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')
            
    torch.save(model.state_dict(), 'finsign_transformer_model.pth')
    np.save('classes.npy', dataset.classes)
    print("Training complete! Model weights and class labels saved.")

if __name__ == "__main__":
    train_model()