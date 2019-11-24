# -*- coding: utf-8 -*-
# @Time    : 1/4/19 11:14 AM
# @Author  : zhoujun
import torch
from torchvision import transforms
import os
import cv2
import time
import numpy as np

from pse import decode as pse_decode


class Pytorch_model:
    def __init__(self, model_path, net, scale, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.scale = scale
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")
        state_dict = torch.load(model_path, map_location=self.device)
        self.net = net.load_state_dict(state_dict)
        print('device:', self.device)

    def predict(self, img: str, long_size: int = 2240):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img), 'file is not exists'
        img = cv2.imread(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        scale = long_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            preds = self.net(tensor)
            preds, boxes_list = pse_decode(preds[0], self.scale)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            # print(scale)
            # preds, boxes_list = decode(preds,num_pred=-1)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            torch.cuda.synchronize()
            t = time.time() - start
        return preds, boxes_list, t


def _get_annotation(label_path):
    boxes = []
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            try:
                label = params[8]
                if label == '*' or label == '###':
                    continue
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            except:
                print('load label failed on {}'.format(label_path))
    return np.array(boxes, dtype=np.float32)


if __name__ == '__main__':
    import config
    from models import PSENet
    import matplotlib.pyplot as plt
    from utils.utils import show_img, draw_bbox

    os.environ['CUDA_VISIBLE_DEVICES'] = str('2')

    model_path = '/content/drive/My Drive/PSENet/PSENet_resnet18.pth'

    # model_path = 'output/psenet_icd2015_new_loss/final.pth'
    image_root = 'Test Set/Image'
    annotation_root = 'Test Set/Annotation'
    
    Images = os.listdir(image_root)
    Labels = os.listdir(annotation_root)
    image_id = Images[np.random.randint(1,len(Images)+1)].split('.jpg')[0]
    
    img_path = image_root+os.sep+'{}.jpg'.format(image_id)
    label_path = annotation_root+os.sep+'{}.txt'.format(image_id)
    
    print('Predicting for image',img_path)
    
    label = _get_annotation(label_path)

    # 初始化网络
    net = PSENet(backbone='resnet18', pretrained=False, result_num=config.n)
    model = Pytorch_model(model_path, net=net, scale=1, gpu_id=0)
    # for i in range(100):
    #     models.predict(img_path)
    preds, boxes_list,t = model.predict(img_path)
    print(boxes_list)
    show_img(preds)
    img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
    cv2.imwrite('result.jpg', img)
    # img = draw_bbox(img, label,color=(0,0,255))
    show_img(img, color=True)

    plt.show()
