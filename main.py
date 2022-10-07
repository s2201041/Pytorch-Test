import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# 訓練データをdatasetsからダウンロード
train_data = torchvision.datasets.FashionMNIST(
              './datasets', train=True, download=True,
              transform=torchvision.transforms.ToTensor())

# テストデータをdatasetsからダウンロード
test_data = torchvision.datasets.FashionMNIST(
              './datasets', train=False, download=True,
              transform=torchvision.transforms.ToTensor())

# データローダーの作成
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True) 

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False) 

# ニューラルネットワークモデルの定義
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU() # 活性化関数 （説明省略）

    def forward(self, x):
        # 順伝播の設定
        # 入力層　→　中間層の全結合
        x = self.fc1(x)
        # 活性化関数 （説明省略）
        x = self.relu(x)
        # 中間層　→　出力層の全結合
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

num_epochs = 100         # 学習を何回、繰り返すか　（エポックと呼ばれる。）
num_batch = 100         # 1度に、何枚の画像を取出すか
learning_rate = 0.001   # 学習率
image_size = 28*28      # 画像の画素数(幅x高さ)

