import cv2
import argparse 
import numpy as np
import torch
from PIL import Image
from retinaface import Retinaface
from utils import utils

retinaface = Retinaface(cuda=torch.cuda.is_available())


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='Face Tools')
    
    # registration_parser = parser.add_subparsers("reg")
    # registration = registration_parser.add_parser()
    
    # parser.add_argument('-r', '--recognize', action="store_true", default=False, help='')
    # parser.add_argument('-o', '--output', default='', help='result image output folder')
    # parser.add_argument('files', nargs='*', help='input files')

    # parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
    #                     type=str, help='Trained state_dict file path to open')
    # parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    # parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    # parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    # parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    # parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    # parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    # parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    # detect.add_argument('-s', '--save_result', action="store_true", default=True, help='show detection results')
    parser.add_argument('files', nargs='*', help='input files')
    args = parser.parse_args()

    feats = {}

    for f in args.files:
        img = np.array(Image.open(f), np.float32)
        result = retinaface.face_detect(img)
        if len(result) > 1:
            print('more than one face was detected')
            continue
        result = result[0]
        # print(result)
        bbox = result[0:4].astype(np.uint)
        conf = result[4]
        lm = result[5:].astype(np.uint).reshape(5, 2)

        # print(bbox)
        # print(conf)
        # print(lm)
        
        cimg = utils.crop_npimage(img, bbox)
        # cv2.imwrite("c1.jpg", cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR))
        
        # 将landmark坐标从全图位置转换为人脸图原点偏移位置
        lm = lm - np.array([int(bbox[0]),int(bbox[1])])
        align, _ = utils.Alignment_1(cimg, lm)
        # cv2.imwrite("c2.jpg", cv2.cvtColor(align, cv2.COLOR_RGB2BGR))

        feat0 = retinaface.face_feature(cimg)
        
        feat = retinaface.face_feature(align)
        print(f'{f}: dist={utils.face_distance(feat0, feat, 0)}')
        feats[f] = feat
        # print(feat)
        # crop(detect(load(f)))

    lk = list(feats.keys())
    for i in range(0, len(lk)):
        for j in range(i + 1, len(lk)):
            xk = lk[i]
            yk = lk[j]
            xv = feats[xk]
            yv = feats[yk]
            dist = utils.face_distance(xv, yv, 0)
            print(f'{xk} vs {yk} = {dist}')

    # print(f'in={args.input}')
    # print(f'out={args.output}')

# r = np.load('model_data/mobilenet_names.npy')
# print(r)

# r = np.load('model_data/mobilenet_face_encoding.npy')
# print(r)

# d1 = {'key1':[5,10], 'key2':[50,100]}
# np.save("feat.npy", d1)

# d2 = np.load("feat.npy", allow_pickle=True)
# print(d2)

#print d1.get('key1')
#print d2.item().get('key2')