function [BinaryTree, HuffCode, BinCode, Codelengths] = TrainHuffTable(qScale, EOB, path, pmf_train)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
     if ~exist('path','var')
        path = struct;
        path(1).name = 'images\sequences\akiyo20_40_RGB';
        path(2).name = 'images\sequences\coastguard20_40_RGB';
        path(3).name = 'images\sequences\foreman20_40_RGB';
        path(4).name = 'images\sequences\news20_40_RGB';
        path(5).name = 'images\sequences\silent20_40_RGB';
     end
     if ~exist('pmf_train','var')
        pmf_train = [];
     end
     if ~exist('EOB','var')
        EOB = 4000;
     end
     MIN_VAL = -1000;
     MAX_VAL = 4000;
     range = -1000 : 4000;
     pmf_train = zeros(1, MAX_VAL - MIN_VAL + 1);
     for i = 1:length(path)
         directory = path(i).name;
         images_dir = dir(fullfile(directory,'*.bmp'));
         num_images = length(images_dir);
         for j = 1:num_images
             train_image=double(imread(fullfile(directory, images_dir(j).name)));
             k_training  = IntraEncode(train_image, qScale, EOB, true);
             pmf_k_training = stats_marg(k_training, range);
             temp = sum((pmf_train + pmf_k_training));
             pmf_train = (pmf_train + pmf_k_training) / temp;
             end
     end
     [BinaryTree, HuffCode, BinCode, Codelengths]=buildHuffman(pmf_train);
end
