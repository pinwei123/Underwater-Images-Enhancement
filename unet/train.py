from unet import *
#from unet_cat import *
#from unet_cross import *
#from unet_cga import *
#from unet_jaffe import *
from earlystop import *
from perceptual_loss import *
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import random
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageFilter
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim, SSIM
from torchmetrics.image import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
"""
from loss_decom_TDN import *
from loss_decom_TDN import *
"""

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class RealBlurDataset(Dataset):
    def __init__(self, file_path, file_path2, transform=None):

        #水下
        self.blur_folder = file_path
        self.gt_folder = file_path2

        
        # 獲取清晰圖像資料夾中的所有文件名
        self.image_pairs = os.listdir(file_path)
        self.image_pairs.sort()  # 可選，確保文件順序一致

        self.transform = transform


    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        gt_path = self.image_pairs[idx]
        blur_path = gt_path
        gt_path = os.path.join(self.gt_folder,gt_path)
        blur_path = os.path.join(self.blur_folder, blur_path)
        gt_image = Image.open(gt_path).convert('RGB')
        blur_image = Image.open(blur_path).convert('RGB')

        if self.transform:
            gt_image = self.transform(gt_image)
            blur_image = self.transform(blur_image)

        return gt_image, blur_image

def get_loader(file_path, file_path2, file_path3, file_path4, batch_size, transform, shuffle=True, num_workers=4):
    dataset_train = RealBlurDataset(file_path=file_path, file_path2=file_path2, transform=transform)
    dataset_test = RealBlurDataset(file_path=file_path3, file_path2=file_path4, transform=transform)
    loader1 = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    loader2 = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=num_workers)
    return loader1, loader2

def show_chart(num_epochs, loss_list, ssim_list, psnr_list, val_loss_list, val_ssim_list, val_psnr_list):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss_list, 'b', label='Train Loss')
    plt.plot(epochs, val_loss_list, 'r', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, ssim_list, 'b', label='Train SSIM')
    plt.plot(epochs, val_ssim_list, 'r', label='Val SSIM')
    plt.title('SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, psnr_list, 'b', label='Train PSNR')
    plt.plot(epochs, val_psnr_list, 'r', label='Val PSNR')
    plt.title('PSNR (dB)')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()

    plt.savefig('/root/notebooks/unet/train_wave.png')


if __name__ == "__main__":

    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    to_pil = transforms.ToPILImage()
    train_loader, test_loader = get_loader('/root/notebooks/LSUI_400_official/trainA', '/root/notebooks/LSUI_400_official/trainB', '/root/notebooks/LSUI_400_official/testA',                
                                           '/root/notebooks/LSUI_400_official/testB', batch_size= 7, transform=data_transforms)
    device = torch.device('cuda')   
    model = UNet(in_channels=3, out_channels=3).to(device)
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    psnr =  PeakSignalNoiseRatio().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=15, delta=0, path='/root/notebooks/unet/train_weight/weight_unet_tdn_2.ph', verbose=True )
    num_epochs = 300

#========================================================== evaluation ========================================================
    loss_list = []
    ssim_list = []
    psnr_list = []
    val_loss_list = []
    val_ssim_list = []
    val_psnr_list = []

    for epoch in range(num_epochs):

        model.train()
        
        epoch_loss = 0.0
        epoch_ssim = 0.0
        epoch_psnr = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for i, (gt_image, blur_image) in enumerate(progress_bar):
            # 移至 GPU
            gt_image = gt_image.to(device)
            gt_image2 = F.interpolate(gt_image, scale_factor=0.5, mode='bilinear')
            gt_image3 = F.interpolate(gt_image, scale_factor=0.25, mode='bilinear')
            blur_image = blur_image.to(device)           
            optimizer.zero_grad()
            outputs, outputs2, outputs3 = model(blur_image)
            
            psnr_value = psnr(outputs, gt_image)
            ssim_value = ssim(outputs, gt_image)
            ssim_value2 = ssim(outputs2, gt_image2)
            ssim_value3 = ssim(outputs3, gt_image3)
            loss =  ( 0.5 * (1 - ssim_value) ) + ( 0.5 * criterion(outputs, gt_image) )
            loss2 =  ( 0.5 * (1 - ssim_value2) ) + ( 0.5 * criterion(outputs2, gt_image2) )
            loss3 =  ( 0.5 * (1 - ssim_value3) ) + ( 0.5 * criterion(outputs3, gt_image3) )
            loss = 0.3*(loss + loss2 + loss3)
            """
            I = F.interpolate(gt_image, size=(32, 32), mode='bilinear', align_corners=False)
            A_gt = estimate_background_light(I)  # 背景光 (B, 1, 1, 1)
            t_gt = estimate_transmission(I, A_gt)  # 透射率 (B, 1, H, W)
            loss_t = criterion(t, t_gt)
            loss_A = criterion(A, A_gt)
            loss = loss + 0.1*loss_t + 0.1*loss_A
            """
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ssim += ssim_value.item()
            epoch_psnr += psnr_value.item()

        epoch_loss = epoch_loss / len(train_loader)
        epoch_ssim = epoch_ssim / len(train_loader)
        epoch_psnr = epoch_psnr / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: { epoch_loss:.4f}, SSIM: {epoch_ssim:.4f}, PSNR: {epoch_psnr:.4f} dB')
        # 計算平均值
        loss_list.append(epoch_loss)
        ssim_list.append(epoch_ssim)
        psnr_list.append(epoch_psnr)

#========================================================== evaluation ========================================================
        model.eval()
        
        val_loss = 0.0
        val_ssim = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for i, (val_gt, val_blur) in enumerate(test_loader):
                # 移至 GPU
                val_gt = val_gt.to(device)
                val_gt2 = F.interpolate(val_gt, scale_factor=0.5, mode='bilinear')
                val_gt3 = F.interpolate(val_gt, scale_factor=0.25, mode='bilinear')
                val_blur = val_blur.to(device)
                val_out, val_out2, val_out3 = model(val_blur)

                psnr_eval = psnr(val_out, val_gt)
                ssim_eval = ssim(val_out, val_gt)
                ssim_eval2 = ssim(val_out2, val_gt2)
                ssim_eval3 = ssim(val_out3, val_gt3)
                loss_eval =  ( 0.5 * (1 - ssim_eval) ) + ( 0.5 * criterion(val_out, val_gt) )
                loss_eval2 =  ( 0.5 * (1 - ssim_eval2) ) + ( 0.5 * criterion(val_out2, val_gt2) )
                loss_eval3 =  ( 0.5 * (1 - ssim_eval3) ) + ( 0.5 * criterion(val_out3, val_gt3) )
                loss_eval = 0.3*(loss_eval + loss_eval2 + loss_eval3)
                """
                I = F.interpolate(val_gt, size=(32, 32), mode='bilinear', align_corners=False)
                A_gt = estimate_background_light(I)  # 背景光 (B, 1, 1, 1)
                t_gt = estimate_transmission(I, A_gt)  # 透射率 (B, 1, H, W)
                loss_t = criterion(t, t_gt)
                loss_A = criterion(A, A_gt)
                loss_eval = loss_eval + 0.1*loss_t + 0.1*loss_A
                """
                val_loss += loss_eval.item()
                val_ssim += ssim_eval.item()
                val_psnr += psnr_eval.item()
                
        val_loss = val_loss / len(test_loader)
        val_ssim = val_ssim / len(test_loader)
        val_psnr = val_psnr / len(test_loader)
        val_loss_list.append(val_loss)
        val_ssim_list.append(val_ssim)
        val_psnr_list.append(val_psnr)
        print(f'Epoch [{epoch+1}/{num_epochs}], Val_Loss: { val_loss:.4f}, Val_SSIM: {val_ssim:.4f}, Val_PSNR: {val_psnr:.4f} dB')
        # 調整learning rate
        scheduler.step(val_loss) 

        early_stopping(val_loss,  model)
        # 如果 early_stop 為 True，則停止訓練
        
        if early_stopping.early_stop:
            print("Early stopping") 
            break
        
    show_chart( epoch+1, loss_list, ssim_list, psnr_list, val_loss_list, val_ssim_list, val_psnr_list)


