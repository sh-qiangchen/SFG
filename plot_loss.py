import matplotlib.pyplot as plt

num_epochs  = 5
epochs = list(range(1, num_epochs + 1))  # 假设 num_epochs 是您的总 epoch 数量
train_losses = [0.8, 0.5, 0.3, 0.2, 0.15]  # 示例训练损失
val_losses = [0.9, 0.6, 0.4, 0.3, 0.2]    # 示例验证损失


plt.figure(figsize=(10, 6))

# plt.plot(epochs, train_losses, label='Training Loss', marker='o')
plt.plot(epochs, train_losses, label='Training Loss')


plt.plot(epochs, val_losses, label='Validation Loss', marker='s')


plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()