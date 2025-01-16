import os
import sys
from PIL import Image
import torchvision.transforms as transforms
import torch
from decoder import Decoder_sigmoid
from hidden.models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]

class Params():
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int,
                attenuation:str, scale_channels:bool, scaling_i:float, scaling_w:float):
        # encoder and decoder parameters
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        # attenuation parameters
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

params = Params(
    encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
    attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
)

decoder = HiddenDecoder(num_blocks=params.decoder_depth, 
        num_bits=params.num_bits, 
        channels=params.decoder_channels
    )
ckpt_path = "./hidden/ckpts/hidden_replicate.pth"
state_dict = torch.load(ckpt_path, map_location='cpu')['encoder_decoder']
encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
decoder.load_state_dict(decoder_state_dict)

decoder = decoder.to(device).eval()



# 设置图像转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
])

# 获取命令行传入的参数（文件夹路径）
input_folder = sys.argv[1]  # 第一个参数是文件夹路径
msg_ori = torch.Tensor(str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0).to(device)
# output_folder = 'path_to_your_output_folder'  # 设置输出文件夹路径


# 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)

total_cnt = 0
match_cnt = 0

input_path = os.path.join(input_folder, 'train/ours_35000/renders')
# 遍历输入文件夹中的所有PNG文件
for filename in os.listdir(input_path):
    if filename.endswith('.png'):
        # 读取图像并应用转换
        image_path = os.path.join(input_path, filename)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # 添加批次维度并移动到指定设备

        total_cnt += 1
        with torch.no_grad():
            output = decoder(image)
            decoded_msg = output > 0
            accs = (~torch.logical_xor(decoded_msg, msg_ori)) # b k -> b k
            if (accs.sum().item() / params.num_bits) > 0.1:
            # if torch.equal(decoded_msg, msg_ori):
                match_cnt += 1
                print(f"Message: {msg2str(msg_ori.squeeze(0).cpu().numpy())}")
                print(f"Decoded: {msg2str(decoded_msg.squeeze(0).cpu().numpy())}")
                print(f"Bit Accuracy: {accs.sum().item() / params.num_bits}")
            
        # 使用Decoder模型处理图像
        # if total_cnt%10 == 0:
        #     with torch.no_grad():
        #         output = decoder(image)
        #         print(output.round())
            #if output == message:
                #match_cnt += 1
        # 保存处理后的张量到输出文件夹
        # output_filename = os.path.splitext(filename)[0] + '_output.pt'
        # output_path = os.path.join(output_folder, output_filename)
        # torch.save(output, output_path)

print('total images: ', total_cnt )
print('correct decode: ', match_cnt)