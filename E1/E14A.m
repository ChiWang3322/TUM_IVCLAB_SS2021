% image read
I_lena = double(imread('lena.tif'));
I_sail = double(imread('sail.tif'));
% Wrap Round
% YOUR CODE HERE
I_lena=padarray(I_lena,[4 4],'replicate','both');
I_sail=padarray(I_sail,[4 4],'replicate','both');
% Resample(subsample)
% YOUR CODE HERE
I_lena_re(:,:,1)=permute(resample( permute(resample(I_lena(:,:,1),1,2,3),[2,1,3]),1,2,3),[2,1,3]);
I_lena_re(:,:,2)=permute(resample(permute(resample(I_lena(:,:,2),1,2,3),[2,1,3]),1,2,3),[2,1,3]);
I_lena_re(:,:,3)=permute(resample(permute(resample(I_lena(:,:,3),1,2,3),[2,1,3]),1,2,3),[2,1,3]);
I_sail_re(:,:,1)=permute(resample(permute(resample(I_sail(:,:,1),1,2,3),[2,1,3]),1,2,3),[2,1,3]);
I_sail_re(:,:,2)=permute(resample(permute(resample(I_sail(:,:,2),1,2,3),[2,1,3]),1,2,3),[2,1,3]);
I_sail_re(:,:,3)=permute(resample(permute(resample(I_sail(:,:,3),1,2,3),[2,1,3]),1,2,3),[2,1,3]);
I_lena=I_lena_re;
I_sail=I_sail_re;
% Crop Back
% YOUR CODE HERE
s1=size(I_lena);
s2=size(I_sail);
I_lena=I_lena(3:s1(1)-2,3:s1(2)-2,:);
I_sail=I_sail(3:s2(1)-2,3:s2(2)-2,:);
% Wrap Round
% YOUR CODE HERE
I_lena=padarray(I_lena,[2 2],'replicate','both');
I_sail=padarray(I_sail,[2 2],'replicate','both');
% Resample (upsample)
% YOUR CODE HERE
I_lena_re2(:,:,1)=permute(resample(permute(resample(I_lena(:,:,1),2,1,3),[2,1,3]),2,1,3),[2,1,3]);
I_lena_re2(:,:,2)=permute(resample(permute(resample(I_lena(:,:,2),2,1,3),[2,1,3]),2,1,3),[2,1,3]);
I_lena_re2(:,:,3)=permute(resample(permute(resample(I_lena(:,:,3),2,1,3),[2,1,3]),2,1,3),[2,1,3]);
I_sail_re2(:,:,1)=permute(resample(permute(resample(I_sail(:,:,1),2,1,3),[2,1,3]),2,1,3),[2,1,3]);
I_sail_re2(:,:,2)=permute(resample(permute(resample(I_sail(:,:,2),2,1,3),[2,1,3]),2,1,3),[2,1,3]);
I_sail_re2(:,:,3)=permute(resample(permute(resample(I_sail(:,:,3),2,1,3),[2,1,3]),2,1,3),[2,1,3]);
I_lena=I_lena_re2;
I_sail=I_sail_re2;
% Crop back
% YOUR CODE HERE
s11=size(I_lena);
s22=size(I_sail);
I_rec_lena=I_lena(5:s11(1)-4,5:s11(2)-4,:);
I_rec_sail=I_sail(5:s22(1)-4,5:s22(2)-4,:);
I_lena3 = double(imread('lena.tif'));
I_sail3 = double(imread('sail.tif'));
% Distortion Analysis
fprintf("%d\n",size(I_rec_lena))
fprintf("%d\n",size(I_rec_sail))
PSNR_lena        = calcPSNR(I_lena3, I_rec_lena);
PSNR_sail        = calcPSNR(I_sail3, I_rec_sail);
fprintf('PSNR lena subsampling = %.3f dB\n', PSNR_lena)
fprintf('PSNR sail subsampling = %.3f dB\n', PSNR_sail)

%display
subplot(221)
imshow(uint8(I_lena3))
title("lena, original")
subplot(222)
imshow(uint8(I_rec_lena))
title("lena, reconstruction")
subplot(223)
imshow(uint8(I_sail3))
title("sail, orignal")
subplot(224)
imshow(uint8(I_rec_sail))
title("sail, reconstruction")
% put all the sub-functions called in your script here
function PSNR = calcPSNR(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
%
% Output        : PSNR     (Peak Signal to Noise Ratio)
% YOUR CODE HERE
% call calcMSE to calculate MSE
PSNR=10*log10((2^8-1).^2/calcMSE(Image, recImage));
end

function MSE = calcMSE(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
% YOUR CODE HERE
%Image=double(Image);
%recImage=double(recImage);
a=sum(sum(sum((Image-recImage).^2)));
b=prod(size(Image));
MSE=a/b;
end