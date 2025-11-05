from restormer import *
from TDN_network import *
from perceptual_loss import *
import random
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from pytorch_msssim import ssim, SSIM
from torchmetrics.image import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
from torchsummary import summary
from loss import AFFTLoss

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class RealBlurDataset(Dataset):
    def __init__(self, file_path, file_path2, transform=None, apply_augmentation_prob=0.5):

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
    loader2 = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader1, loader2

def show_image(image_1, image_2, image_3, i ):

    output_dir = os.path.join("/root/notebooks/unet/outputs/TDN_block", f'{i}_compare.png')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1) 
    plt.imshow(image_1)
    plt.title('gt')
    plt.axis('off')
   
    plt.subplot(1, 3, 2)
    plt.imshow(image_2)
    plt.title('output')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image_3)
    plt.title('blur')
    plt.axis('off')
    plt.savefig(output_dir)


def tensor_to_image(tensor, file_path):
    single_image = tensor.detach().cpu().numpy()
    single_image = np.transpose(single_image, (1, 2, 0))  # 轉換為 (H, W, C)
    single_image = (single_image * 255).clip(0, 255).astype(np.uint8)  # 調整範圍並轉換為 uint8
    image = Image.fromarray(single_image)
    #image.save(file_path)
    return image

if __name__ == "__main__":
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    to_pil = transforms.ToPILImage()
    train_loader, test_loader = get_loader('/root/notebooks/LSUI_400_official/trainA', '/root/notebooks/LSUI_400_official/trainB', '/root/notebooks/LSUI_400_official/testA',                
                                           '/root/notebooks/LSUI_400_official/testB', batch_size= 1, transform=data_transforms)
    device = torch.device('cuda')    

    """
    model = Restormer(  inp_channels=3, 
                        out_channels=3, 
                        dim = 48,
                        num_blocks = [4,6,6,8], 
                        num_refinement_blocks = 4,
                        heads = [1,2,4,8],
                        ffn_expansion_factor = 2.66,
                        bias = False,
                        LayerNorm_type = 'WithBias',   
                        dual_pixel_task = False ).to(device)
    """
    model = TDN( inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 2, 2, 4], #2，3，3，4
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias' ).to(device)
    
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    psnr =  PeakSignalNoiseRatio().to(device)
    criterion = torch.nn.MSELoss()

    try:
        model.load_state_dict(torch.load('/root/notebooks/unet/train_weight/weight_TDN_block.ph'))
        print("Loaded saved model weights.")
    except FileNotFoundError:
        print("No saved model weights found. Training from scratch.") 

#========================================================== evaluation ========================================================
    model.eval()
    
    val_loss = 0.0
    val_ssim = 0.0
    val_psnr = 0.0
    with torch.no_grad():
        for i, (val_gt, val_blur) in enumerate(test_loader):
            # 移至 GPU
            val_gt = val_gt.to(device)
            val_blur = val_blur.to(device)
            val_out = model(val_blur)

            psnr_eval = psnr(val_out, val_gt)
            ssim_eval = ssim(val_out, val_gt)
            loss_eval =  ( 0.5 * (1 - ssim_eval) ) + ( 0.5 * criterion(val_out, val_gt) )

            val_loss += loss_eval.item()
            val_ssim += ssim_eval.item()
            val_psnr += psnr_eval.item()

            if (i + 1) % 26 == 0:  # 每10个batch保存一次图片
                output_dir = "/root/notebooks/unet/outputs"
                
                output_image_path = os.path.join(output_dir, f'{i+1}_output.png')
                output_image = tensor_to_image(val_out[0], output_image_path )
            
                blur_image_path = os.path.join(output_dir, f'{i+1}_blur.png')
                blur_image = tensor_to_image(val_blur[0], blur_image_path )
    
                gt_image_path = os.path.join(output_dir, f'{i+1}_gt.png')
                gt_image = tensor_to_image(val_gt[0], gt_image_path )
    
                show_image( gt_image, output_image, blur_image, i+1 )
            
    
    val_loss = val_loss / len(test_loader)
    val_ssim = val_ssim / len(test_loader)
    val_psnr = val_psnr / len(test_loader)
    print(f' Test_Loss: { val_loss:.4f}, Test_SSIM: {val_ssim:.4f}, Test_PSNR: {val_psnr:.4f} dB')





 




