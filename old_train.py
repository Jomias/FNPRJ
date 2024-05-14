import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from numba.experimental import jitclass
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fen import parse_fen, start_position


def fen_to_bit_vector(fen):
    parse_pos = parse_fen(fen)
    return parse_pos.extract_bit_vector()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(17, 832)
        self.fc2 = nn.Linear(832, 416)
        self.fc3 = nn.Linear(416, 208)
        self.fc4 = nn.Linear(208, 104)
        self.fc5 = nn.Linear(104, 1)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class ChessDataset(Dataset):
    def __init__(self, data_frame):
        self.fens = torch.from_numpy(np.array([*map(fen_to_bit_vector, data_frame["FEN"])], dtype=np.float32))
        self.evals = torch.Tensor([[x] for x in data_frame["Evaluation"]])
        self._len = len(self.evals)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.fens[index], self.evals[index]


def eval_to_int(evaluation):
    try:
        res = int(evaluation)
    except ValueError:
        res = 10000 if evaluation[1] == '+' else -10000
    return int(res / 100)


def AdamW_main():
    MAX_DATA = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))

    print("Preparing Training Data...")
    train_data = pd.read_csv("data/chessData.csv")
    train_data = train_data[:MAX_DATA]
    train_data["Evaluation"] = train_data["Evaluation"].map(eval_to_int)
    trainset = ChessDataset(train_data)

    print("Preparing Test Data...")
    test_data = pd.read_csv("data/tactic_evals.csv")
    test_data = test_data[:MAX_DATA]
    test_data["Evaluation"] = test_data["Evaluation"].map(eval_to_int)
    test_set = ChessDataset(test_data)

    batch_size = 10

    print("Converting to pytorch Dataset...")

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.01)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                # denominator for loss should represent the number of positions evaluated
                # independent of the batch size
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (2000 * len(labels))))
                running_loss = 0.0

    print('Finished Training')

    PATH = 'adam_model_0.pth'
    torch.save(net.state_dict(), PATH)

    print('Evaluating model')

    count = 0
    total_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # print("Correct eval: {}, Predicted eval: {}, loss: {}".format(labels, outputs, loss))

            # count should represent the number of positions evaluated
            # independent of the batch size
            count += len(labels)
            total_loss += loss
            if count % 10000 == 0:
                print('Average error of the model on the {} tactics positions is {}'.format(count, loss / count))
    # print('Average error of the model on the {} tactics positions is {}'.format(count, loss/count))


def load_model(path):
    m = Net()
    m.load_state_dict(torch.load(path))
    m.eval()
    return m


def predict_evaluation(model, vector):
    with torch.no_grad():
        bit_vector_tensor = torch.from_numpy(np.array([vector], dtype=np.float32))
        output = model(bit_vector_tensor)
        return output.item()  # Assuming your model outputs a single evaluation score


if __name__ == "__main__":
    # AdamW_main()
    model = load_model("adam_model_0.pth")

