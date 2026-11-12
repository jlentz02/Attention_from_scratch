import torch
from data_loading import load_data
from basic_multihead_transformer import MultiheadAttentionNN
import torch.optim as optim




def train_epoch(epoch):
    running_loss = 0

    for i, data in enumerate(dataloader):
        inputs, labels = data

        optimizer.zero_grad

        outputs = model(inputs)

        CLS = outputs[:, 0]

        loss = loss_fn(CLS, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss

        if i%10 == 9:
            print(f"Running loss is: {running_loss/(i+1)}")
    
    return running_loss/1000


batch_size = 16
dataloader = load_data(batch_size, 1000)

model = MultiheadAttentionNN(
    batch_size=batch_size,
    patch_size=4,
    output_dim=10,
    model_dim=64,
    key_dim=64,
    value_dim=64,
    heads=8
)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(1):
    epoch_loss = train_epoch(epoch)
    print(f"Epoch loss: {epoch_loss}")
