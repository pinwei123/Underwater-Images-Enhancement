import torch
import os


class EarlyStopping:
    def __init__(self, patience, delta, path, verbose=False):
        self.patience = patience  # 在多少個 epoch 沒有提升後停止訓練
        self.delta = delta  # 判斷提升的最小變化
        self.best_loss = None
        self.counter = 0  # 計數器，用於記錄沒有提升的 epoch 數
        self.early_stop = False  # 是否停止訓練的標誌
        self.path = path  # 模型保存路徑
        self.verbose = verbose  # 是否打印信息

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss  # 第一次設置最佳損失
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss  # 更新最佳損失
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # 重置計數器
        else:
            self.counter += 1  # 沒有提升，計數器加 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # 停止訓練

    def save_checkpoint(self, val_loss, model):
        '''保存模型'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


 