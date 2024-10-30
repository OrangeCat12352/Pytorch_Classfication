import csv

import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
from models.swiftformer import *


classes = ('Real', 'Fake')
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = SwiftFormer_L3()
#model.load_state_dict(torch.load('weight/best.pth'))
model = torch.load(r'C:\Users\Chen Xuhui\Desktop\Pytorch_Classfication\weight\Swiftformer\best.pth')
model.eval()
model.to(DEVICE)

path = 'DataSet/test/'
results = []
testList = os.listdir(path)
for file in testList:
    img = Image.open(path + file).convert('RGB')
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    # 将图片名和预测结果加入列表
    results.append((file.split('.')[0], int(pred)))  # 转换为整数
    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))

# 按字典序排序（默认已经排序）
results.sort(key=lambda x: x[0])
# 保存结果到 CSV 文件
output_csv = './cla_pre.csv'
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for row in results:
        writer.writerow(row)  # 每行包含图片名和预测结果

print(f"Results saved to {output_csv}")
