import os
import cv2
import paddle
from paddle_msssim import ssim, ms_ssim


def file_name(file_dir):
    img_path_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            img_path_list.append((os.path.join(root, file), file))
    return img_path_list


def imread(img_path):
    img = cv2.imread(img_path)
    return paddle.to_tensor(img.transpose(2, 0, 1)[None, ...], dtype=paddle.float32)


if __name__ == '__main__':
    file_dir = './results/animal_lofgan/test_for_fid'  # 伪造图像路径
    target_dir = 'datasets/for_fid/animal'  # 真实图像路径

    img_path_list = file_name(file_dir)
    target_path_list = file_name(target_dir)
    d = 0
    for i in range(img_path_list.__len__()):
        (img_path, img_name) = img_path_list[i]
        (target_path, target_name) = target_path_list[i]
        print(img_path)
        print(target_path)
        fake = imread(img_path)
        real = imread(target_path)

        distance = ssim(real, fake).cpu().numpy()
        print(distance)
        d += distance
print('average ssim')
print(d / img_path_list.__len__())

