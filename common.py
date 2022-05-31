import torch
from torch import optim
from torch.utils import data
from torch.nn import functional
from torchvision import datasets, transforms
import os

BATCH_SIZE = 64
DATASET_DIRECTORY = 'dataset'
MODEL_DIRECTORY = 'models'


def accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    return top_pred.eq(y.view_as(top_pred)).sum().float() / y.shape[0]


def _train(model, train_loader, optimizer):
    loss_sum = 0
    acc_sum = 0
    model.train()

    for (x, y) in train_loader:
        optimizer.zero_grad()
        y_pred, _ = model(x)

        loss = functional.nll_loss(y_pred, y, reduction='mean')
        acc = accuracy(y_pred, y)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        acc_sum += acc.item()

    train_loader_len = len(train_loader)
    return loss_sum / train_loader_len, acc_sum / train_loader_len


def evaluate(model, iterator):
    loss_sum = 0
    acc_sum = 0
    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            y_pred, _ = model(x)
            loss_sum += functional.nll_loss(y_pred, y, reduction='sum').item()
            acc_sum += accuracy(y_pred, y).item()

    test_loader_len = len(iterator)
    return loss_sum / test_loader_len, acc_sum / test_loader_len


def get_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    download = not os.path.exists(os.path.join(DATASET_DIRECTORY, 'cifar-10-python.tar.gz'))

    train_data = datasets.CIFAR10(DATASET_DIRECTORY, train=True, download=download, transform=transform_train)
    test_data = datasets.CIFAR10(DATASET_DIRECTORY, train=False, transform=transform_test)

    print(f'Loaded data: {len(train_data)} train and {len(test_data)} test examples')

    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE)

    return train_loader, test_loader


def train(model, file_name, epochs=50):

    file_path = os.path.join(MODEL_DIRECTORY, f'{file_name}.pt')

    train_loader, test_loader = get_loaders()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    best_test_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = _train(model, train_loader, optimizer)
        test_loss, test_acc = evaluate(model, test_loader)

        if test_acc > best_test_acc:
            print(f'Saving file {file_path}, test_accuracy ({test_acc}) > best_test_accuracy({best_test_acc})')
            torch.save(model.state_dict(), file_path)
            best_test_acc = test_acc

        if epoch % 10 == 9:
            print(
                f'{epoch:02}: train: loss {train_loss:.3f}, acc {train_acc * 100:.2f}%, ' +
                    f'test: loss {test_loss:.3f}, acc {test_acc * 100:.2f}%, best test acc: {best_test_acc * 100:.2f}%'
            )
        else:
            print(f'{epoch:02}, ', end='')
