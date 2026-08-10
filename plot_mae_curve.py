import json
import matplotlib.pyplot as plt

with open('outputs/efficientnet_b3/fold_4/history.json') as f:
    history = json.load(f)

plt.figure(figsize=(8,5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('docs/assets/loss_curve_mae.png')
plt.close()
print('Saved: docs/assets/loss_curve_mae.png')
