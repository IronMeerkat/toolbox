import torch
from torch import nn
from sklearn.model_selection import train_test_split as _train_test_split
from typing import Any
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected device is {DEVICE}")
torch.set_default_device(DEVICE)

def fmt(x):
    return f"{int(x*100)}%"

def improvement_ratio(df: pd.DataFrame, model_accuracy: float):
    vals = get_value_distribution(df).to_numpy() /100
    known = sum(vals ** 2)
    monkey = 1/len(vals)
    highest = vals.max()
    print(f"Random known : Baseline: {fmt(known)}, ratio {fmt(model_accuracy/known)}")
    print(f"Monkey pick  : Baseline: {fmt(monkey)}, ratio {fmt(model_accuracy/monkey)}")
    print(f"Highest pick : Baseline: {fmt(highest)}, ratio {fmt(model_accuracy/highest)}")


def torchify(arr) -> torch.Tensor:
    return torch.from_numpy(arr).type(torch.float32).to(DEVICE)

def get_value_distribution(df: pd.DataFrame):
    return df['target'].value_counts() / df['target'].count() * 100

def train_test_split(X, y):
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_size=0.8)
    X_train = X_train.float()
    X_test = X_test.float()
    y_train = y_train.float()
    y_test = y_test.float()
    return X_train, X_test, y_train, y_test

def get_features_and_labels(df: pd.DataFrame, n_components: int = None, is_classifier=True):

    raw_X = df.drop('target', axis=1).to_numpy()
    y = df['target'].to_numpy()

    raw_X = StandardScaler().fit_transform(raw_X)
    y = y if is_classifier else StandardScaler().fit_transform(y.reshape(-1, 1))

    y = torchify(y).squeeze()
    X = PCA(n_components=n_components).\
        fit_transform(raw_X) if n_components else raw_X
    X = torchify(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def accuracy_fn(y_true, y_pred):
    correct = (y_true == y_pred).type(torch.float32)
    return float(correct.mean()) * 100

def get_important_columns(pd_mat: pd.DataFrame, filter=0):
    """To see visually if any columns stand out as worth or not worth keeping. """
    normalizer = StandardScaler()
    mat = normalizer.fit_transform(pd_mat.to_numpy())
    cov_mat = mat.T @ mat
    cov_mat = (cov_mat + cov_mat.T) / 2
    eigs = np.linalg.eigvals(cov_mat)
    eigs = np.absolute(eigs)
    eigs = eigs[eigs >= 10 ** filter -1]
    plt.figure(figsize=(5, 3))
    plt.bar(range(len(eigs)), eigs)
    plt.yscale('log')
    plt.xlabel('Index of Eigenvalue')
    plt.ylabel('Eigenvalue (log scale)')
    plt.title('Eigenvalues of the Covariance Matrix')
    plt.show()

def plot_loss_curves(results):
    """Plots training curves of a results dictionary."""
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

class TrinomialRegression(nn.Module):

    def __init__(self, num_features, num_outputs=1):
        super(TrinomialRegression, self).__init__()
        self.l1 = nn.Linear(num_features, num_outputs)
        self.l2 = nn.Linear(num_features, num_outputs)
        self.l3 = nn.Linear(num_features, num_outputs)

    def forward(self, x):
        out1 = self.l1(x)
        out2 = self.l2(x)
        out3 = self.l3(x)

        return (out1 + torch.pow(out2, 2) + torch.pow(out3, 3)).squeeze()

class Loop:

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    def __init__(self, model, loss_fn=nn.CrossEntropyLoss,
                 optimizer=torch.optim.SGD, accuracy_fn=accuracy_fn, lr=0.1) -> None:
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []
        self.model = model
        self.loss_fn = loss_fn()
        self.optimizer = optimizer(params=model.parameters(), lr=lr)
        self.accuracy_fn = accuracy_fn

    def train(self, X_train, y_train):
        self.model.train()
        train_loss, train_acc = 0, 0

        X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE).long()
        y_logits = self.model(X_train)
        loss = self.loss_fn(y_logits, y_train)
        train_loss += loss
        train_acc += self.accuracy_fn(y_true=y_train, y_pred=y_logits.argmax(dim=1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_loss.append(float(train_loss))
        self.train_acc.append(train_acc)
        return train_loss, train_acc


    def test(self, X_test, y_test):
        self.model.eval()
        test_loss, test_acc = 0, 0

        with torch.inference_mode():

            X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE).long()
            y_logits = self.model(X_test)
            test_loss += self.loss_fn(y_logits, y_test)
            test_acc += self.accuracy_fn(y_true=y_test, y_pred=y_logits.argmax(dim=1))

            self.test_loss.append(float(test_loss))
            self.test_acc.append(test_acc)
            return test_loss, test_acc

    def __call__(self, epochs, X_train, X_test, y_train, y_test) -> Any:
        epoch_mod = epochs / 10
        for epoch in range(epochs):

            train_loss, train_acc = self.train(X_train, y_train)
            test_loss, test_acc = self.test(X_test, y_test)
            if (epoch + 1) % epoch_mod == 0:
                print(f"Epoch: {epoch}: ",
                    f"Train loss: {train_loss:.4f} ",
                    f"| Train acc: {train_acc:.4f} ",
                    f"| Test loss: {test_loss:.4f} ",
                    f"|  Test acc: {test_acc:.4f}")

    def plot_loss_curves(self):
        plot_loss_curves({
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "test_loss": self.test_loss,
            "test_acc": self.test_acc,
        })
