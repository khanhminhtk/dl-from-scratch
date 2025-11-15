import torch
from model.multilayer_perceptron import MultiLayerPerceptron
from optimation.optimizers import Adam, RMSprop    

drive = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
batch_size, input_dim, output_dim = 16, 10, 3
X = torch.randn(batch_size, input_dim).to(drive)
target_idx = torch.randint(0, output_dim, (batch_size,), device=drive)
y_true = torch.nn.functional.one_hot(target_idx, num_classes=output_dim).float()

mlp = MultiLayerPerceptron(drive=drive, input_dim=input_dim, output_dim=output_dim)
# optimizer = Adam(mlp.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
optimizer = RMSprop(mlp.parameters(), lr=1e-2, alpha=0.99, eps=1e-8)


for step in range(300):
    loss, preds = mlp.train_step(X, y_true, optimizer)
    print(f"step={step} loss={loss.item():.4f}")