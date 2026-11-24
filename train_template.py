import torch
from data_loading import load_data
from basic_multihead_transformer import MultiheadAttentionNN
import torch.optim as optim

#The purpose of this file was to organize the training loop before I moved it 
#into a Jupyter notebook where it is easier to experiment with things


def train_epoch(epoch):
    running_loss = 0
    running_acc = 0
    for i, data in enumerate(dataloader_train):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        CLS = outputs[:, 0, :]

        with torch.no_grad():
            preds = torch.argmax(CLS, dim=1)
            correct = (preds == labels)
            accuracy = correct.float().sum()
            running_acc += accuracy


        loss = loss_fn(CLS, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    
    return running_loss, running_acc

def test_model():
    running_acc = 0
    print(len(dataloader_test))
    for i, data in enumerate(dataloader_test):
        inputs, labels = data

        outputs = model(inputs)

        CLS = outputs[:, 0, :]

        with torch.no_grad():
            preds = torch.argmax(CLS, dim=1)
            correct = (preds == labels)
            accuracy = correct.float().sum()
            running_acc += accuracy

    return running_acc


batch_size = 32
data_size = 10000
tts_ratio = 0.8
train_size = data_size*tts_ratio
test_size = data_size*(1-tts_ratio)
dataloader_train, dataloader_test = load_data(batch_size, data_size, tts_ratio=tts_ratio)

model = MultiheadAttentionNN(
    batch_size=batch_size,
    patch_size=7,
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
for epoch in range(5):
    epoch_loss, accuracy = train_epoch(epoch)
    print(f"Epoch {epoch+1} | Loss: {epoch_loss/(train_size//batch_size):.4f} | Accuracy: {accuracy/train_size:.4f}")

test_accuracy = test_model()

print(f"Test accuracy: {test_accuracy/test_size:.4f}")




    
