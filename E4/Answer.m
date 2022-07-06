function dst = IntraEncode(image, qScale)
% Function Name : IntraEncode.m
% Input : image (Original RGB Image)
% qScale(quantization scale)
% Output : dst (sequences after zero-run encoding, 1xN)
yuv_im = ictRGB2YCbCr(image);
im_dct = blockproc(yuv_im, [8, 8], @(block_struct) DCT8x8(block_struct.data));
im_quant = blockproc(im_dct, [8, 8], @(block_struct) Quant8x8(block_struct.data, qScale));
for i = 1:3
row_blks_count = 1;
for c = 1 : size(image, 2)/8
for r = 1 : size(image,1)/8
tmp_quant((row_blks_count-1)*8+1 : row_blks_count*8, 1:8, i) = ...
im_quant( (r-1)*8+1 : r*8, (c-1)*8+1 : c*8, i);
row_blks_count = row_blks_count + 1;
end
end
end
im_quant = tmp_quant;
im_zz = blockproc(im_quant, [8, 8], @(block_struct) ZigZag8x8(block_struct.data));
dst = ZeroRunEnc_EoB(im_zz(:));
end

function dst = IntraDecode(image, img_size , qScale)
% Function Name : IntraDecode.m
% Input : image (zero-run encoded image, 1xN)
% img_size (original image size)
% qScale(quantization scale)
% Output : dst (decoded image)
de_zz = ZeroRunDec_EoB(image);
im_de_zz = reshape(de_zz, [img_size(1)*img_size(2), 1, 3]);
de_quant = blockproc(im_de_zz, [64, 1], @(block_struct) DeZigZag8x8(block_struct.data));
for i = 1:3
row_blks_count = 1;
for c = 1 : img_size(2)/8
for r = 1 : img_size(1)/8 
im_de_quant((r-1)*8+1 : r*8, (c-1)*8+1 : c*8, i) = ...
de_quant((row_blks_count-1)*8+1 : row_blks_count*8, 1:8, i);
row_blks_count = row_blks_count + 1;
end
end
end
im_idct = blockproc(im_de_quant, [8, 8], @(block_struct) DeQuant8x8(block_struct.data, qScale));
yuv_dst = blockproc(im_idct, [8, 8], @(block_struct) IDCT8x8(block_struct.data));
dst = ictYCbCr2RGB(yuv_dst);
end