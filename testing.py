#from traning import*


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

model.eval()  # モデルを評価モードにする

loss_sum = 0
correct = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ニューラルネットワークの処理を実施
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss_sum += criterion(outputs, labels)

        # 正解の値を取得
        pred = outputs.argmax(1)
        # 正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_data)}% ({correct}/{len(test_data)})")