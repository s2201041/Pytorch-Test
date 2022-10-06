from main import*

# GPUが使える場合は、GPU使用モードにする。
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# ニューラルネットワークの生成して、GPUにデータを送る
model = NeuralNet().to(device)
# モデルを訓練モードにする
model.train()  

# 損失関数の設定（説明省略）
criterion = nn.CrossEntropyLoss() 
# 最適化手法の設定（説明省略）
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

# 設定したエポック数分、学習する。
for epoch in range(num_epochs): 
    loss_sum = 0

    for inputs, labels in train_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()

        # ニューラルネットワークの処理を実施
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss = criterion(outputs, labels)
        loss_sum += loss

        # 学習
        loss.backward()
        optimizer.step()

    # 学習状況の表示
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")

