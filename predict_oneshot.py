import cv2
import argparse
import numpy as np
from utils import utils

from retinaface import Retinaface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retinaface')

    # parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
    #                     type=str, help='Trained state_dict file path to open')
    # parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    # parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    # parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    # parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    # parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    # parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    # parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    # parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    parser.add_argument('files', nargs='*', help='input files')
    args = parser.parse_args()

    retinaface = Retinaface()

    '''
    predict.py有几个注意点
    1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用cv2.imread打开图片文件进行预测。
    2、如果想要保存，利用cv2.imwrite("img.jpg", r_image)即可保存。
    3、如果想要获得框的坐标，可以进入detect_image函数，读取(b[0], b[1]), (b[2], b[3])这四个值。
    4、如果想要截取下目标，可以利用获取到的(b[0], b[1]), (b[2], b[3])这四个值在原图上利用矩阵的方式进行截取。
    5、在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
    '''
    for i, img in enumerate(args.files):
        image = cv2.imread(img)
        if image is None:
            print('Open Error! Try again!')
        else:
            image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            lms     = retinaface.face_detect(image)
            face    = utils.crop_npimage(np.array(image), lms[:4])
            print(lms)
            r_image = retinaface.detect_image(image)
            r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"out{i}.jpg", r_image)
