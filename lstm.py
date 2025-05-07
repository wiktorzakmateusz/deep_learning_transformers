import pandas as pd

pd.options.display.max_columns = 50
pd.options.display.max_rows = 50
from datetime import datetime
import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

SAMPLE_RATE = 16000
N_MELS = 64
BATCH_SIZE = 1
DATA_DIR = 'C:\\Users\\kinga\\OneDrive\\Dokumenty\\Studia\\Magisterka\\semestr_1\\DeepLearning\\deep_learning_transformers\\audio_data\\audio'
VAL_LIST = 'audio_data/train/validation_list_with_silence_balanced.txt'
TEST_LIST = 'audio_data/train/testing_list_with_silence_balanced.txt'
TRAIN_LIST = 'audio_data/train/training_list_with_silence_balanced.txt'


class AudioTransform:
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform):
        return self.db(self.mel(waveform))


class AudioDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        self.fixed_length = 32  # number of time frames

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(path)

        # Convert to MelSpectrogram
        mel_spec = T.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
        mel_spec = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10, db_multiplier=0, amin=1e-10,
                                                         top_db=80)

        return mel_spec, label


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softAct = nn.Softmax()

    def forward(self, x):
        if len(x.shape) == 4:
            x = x[0]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x.mT, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softAct(out)
        return out

    def train_one_epoch(self, optimizer, train_dl, train_ds, device):
        self.train()
        total_loss, total_correct = 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb[0]
            out = self(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            total_correct += (out.argmax(1) == yb).sum().item()
        return total_loss / len(train_ds), total_correct / len(train_ds)

    def evaluate(self, dl):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        correct = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                out = self(xb)
                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
                correct += (preds == yb).sum().item()
        return correct / len(dl.dataset), all_preds, all_labels

    def train_cnn(self, epochs, data, learning_rate, weight_decay, device):

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.train_accs, self.val_accs = [], []

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(optimizer, data.train_dl, data.train_ds, device)
            val_acc, _, _ = self.evaluate(data.val_dl)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            print(
                f"Epoch {epoch + 1}: Train acc = {train_acc:.4f}, Val acc = {val_acc:.4f} - {datetime.now().strftime('%H:%M:%S')}")


class Data:
    def __init__(self):
        def load_file_paths(file_list_path, base_dir):
            with open(file_list_path, 'r') as f:
                files = [os.path.join(base_dir, line.strip()) for line in f]
            return files

        train_files = load_file_paths(TRAIN_LIST, DATA_DIR)
        val_files = load_file_paths(VAL_LIST, DATA_DIR)
        test_files = load_file_paths(TEST_LIST, DATA_DIR)

        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files

        def map_label(label):
            return label if label in {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                                      'silence'} else 'unknown'

        # Extract and map labels from file paths
        train_labels = [map_label(os.path.basename(os.path.dirname(f))) for f in train_files]
        val_labels = [map_label(os.path.basename(os.path.dirname(f))) for f in val_files]
        test_labels = [map_label(os.path.basename(os.path.dirname(f))) for f in test_files]

        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels

        labels = list(set(train_labels))
        le = LabelEncoder()
        le.fit(labels)

        self.labels = labels
        self.le = le

        # Encode labels
        train_targets = le.fit_transform(train_labels)
        val_targets = le.transform(val_labels)
        test_targets = le.transform(test_labels)

        # Create datasets
        transform = AudioTransform()
        train_ds = AudioDataset(train_files, train_targets, transform=transform)
        val_ds = AudioDataset(val_files, val_targets, transform=transform)
        test_ds = AudioDataset(test_files, test_targets, transform=transform)

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl


def train_models_with_hyperparams():

    data = Data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 10
    results = []
    for learning_rate in [0.001, 0.0005, 0.0001]:
        for weight_decay in [0, 0.0001, 0.0005]:
            print('\nlr = {}, weight_decay = {}'.format(learning_rate, weight_decay))

            model = LSTMClassifier(input_size=64, hidden_size=10, num_layers=4, output_size=12)
            model.train_cnn(epochs=epochs, data=data, learning_rate=learning_rate, weight_decay=weight_decay, device=device)

            test_acc, preds, truths = model.evaluate(data.test_dl)

            print('test accuracy: {}'.format(test_acc))

            results.append([epochs, learning_rate, weight_decay, test_acc])

    results_df = pd.DataFrame(results, columns=['epochs', 'learning_rate', 'weight_decay', 'test_acc'])
    results_df.to_csv('results_lstm.csv', index=False)


def train_final_model():
    epochs = 100
    learning_rate = 0.0005
    weight_decay = 0.0001

    data = Data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(input_size=64, hidden_size=10, num_layers=4, output_size=12)
    model.train_cnn(epochs=epochs, data=data, learning_rate=learning_rate, weight_decay=weight_decay, device=device)

    test_acc, preds, truths = model.evaluate(data.test_dl)
    df = pd.DataFrame({"train": model.train_accs, "val": model.val_accs})
    df.to_csv('train_val_accs_lstm.csv', index=False)

    df = pd.DataFrame({"truth": truths, "pred": preds})
    df.to_csv('truth_prediction_lstm.csv', index=False)

if __name__ == "__main__":
    train_final_model()