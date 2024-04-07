import os
import sys
from PIL import Image
import torchvision.transforms as transforms
import torch
from decoder import Decoder_sigmoid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建Decoder模型实例
decoder = Decoder_sigmoid(decoder_channels=64, decoder_blocks=6, message_length=4).to(device)

# 设置设备为CPU或GPU

decoder.to(device)

# 设置图像转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
])

# 获取命令行传入的参数（文件夹路径）
input_folder = sys.argv[1]  # 第一个参数是文件夹路径
message = sys.argv[2]
# output_folder = 'path_to_your_output_folder'  # 设置输出文件夹路径

ckpt = torch.load(os.path.join(input_folder,'ckpt.tar'))
decoder.load_state_dict(ckpt['decoder_state_dict'])

# 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)

total_cnt = 0
match_cnt = 0

input_path = os.path.join(input_folder, 'train/ours_30000/renders')
# 遍历输入文件夹中的所有PNG文件
for filename in os.listdir(input_path):
    if filename.endswith('.png'):
        # 读取图像并应用转换
        image_path = os.path.join(input_path, filename)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # 添加批次维度并移动到指定设备

        total_cnt += 1
        # 使用Decoder模型处理图像
        with torch.no_grad():
            output = decoder(image)
            print(output)
            if output == message:
                match_cnt += 1
        # 保存处理后的张量到输出文件夹
        # output_filename = os.path.splitext(filename)[0] + '_output.pt'
        # output_path = os.path.join(output_folder, output_filename)
        # torch.save(output, output_path)

print('total images: ', total_cnt )
print('correct decode: ', match_cnt)