function dst = IntraDecode(image, img_size , qScale, EOB, ict)
%  Function Name : IntraDecode.m
%  Input         : image (zero-run encoded image, 1xN)
%                  img_size (original image size)
%                  qScale(quantization scale)
%  Output        : dst   (decoded image)
    image_zzd = ZeroRunDec_EoB(image, EOB);
    num_rows = img_size(1) / 8 * 64;
    num_columns = img_size(2) / 8;
    image_zzd = reshape(image_zzd(:), [num_rows, num_columns * img_size(3)]);   %correct
    [M, N] = size(image_zzd);
    image_dezig = zeros(img_size);  %Correct
    for i = 1:N
        temp = blockproc(image_zzd(:, i), [64, 1], @(block_struct) DeZigZag8x8(block_struct.data));
        current_dim = mod(i, 3);
        if current_dim == 0
            current_dim = 3;
        end
        current_index = floor((i - 1)/3);
        image_dezig(:, current_index*8 + 1: (current_index + 1)*8, current_dim) = temp;
    end
   
    
    image_dequant = blockproc(image_dezig, [8, 8], @(block_struct) DeQuant8x8(block_struct.data, qScale));
    image_IDCT = blockproc(image_dequant, [8, 8], @(block_struct) IDCT8x8(block_struct.data));
    if ict == true
        dst = ictYCbCr2RGB(image_IDCT);
    else
        dst = image_IDCT;
    end
end

function [DC, AC] = IntraEncode(image, qScale, EOB, ict)
%  Function Name : IntraEncode.m
%  Input         : image (Original RGB Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)
    if ict == true
        imageYUV = ictRGB2YCbCr(image);
    else
        imageYUV = image;
    end
    
    %DCT Transform
    [M, N, C] = size(image);
    imageYUV_DCT = blockproc(imageYUV, [8, 8], @(block_struct) DCT8x8(block_struct.data));
    imageYUV_quant = blockproc(imageYUV_DCT, [8, 8], @(block_struct) Quant8x8(block_struct.data, qScale));
    imageYUV_zz = blockproc(imageYUV_quant, [8, 8], @(block_struct) ZigZag8x8(block_struct.data));
    dst = ZeroRunEnc_EoB(imageYUV_zz(:), EOB);
    DC = zeros(1, M * N * C / 64);
    AC = zeros(siez(dst));
    pDC = 1;
    pAC = 1;
    DC(pDC) = dst(1);
    pDC = pDC + 1;
    for i = 2 : len(dst)
        if dst(i - 1) == EOB
            DC(pDC) = dst(i);
            pDC = pDC + 1;
        else
            AC(pAC) = dst(i);
            pAC = pAC + 1;
        end
    end
        
end