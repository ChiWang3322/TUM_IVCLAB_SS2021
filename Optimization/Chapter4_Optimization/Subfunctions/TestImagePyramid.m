clc, clear;
image = double(imread('ForemanSequence\foreman0020.bmp'));
Pyramid = GetImagePyramid(image, 3);
rec_image = ImagePyramid_rec(Pyramid);
