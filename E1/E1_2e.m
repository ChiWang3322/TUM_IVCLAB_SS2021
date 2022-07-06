% Read Image
I = double(imread('monarch.tif'));
%Define kernel
kernel = [1, 2, 1; 2, 4, 2; 1, 2, 1];
%Normalize kernel
kernel = kernel ./ sum(kernel(:));  
%Visualize kernel
F_kernel = fftshift(fft2(kernel));
figure;
imshow(abs(F_kernel))
% without prefiltering
% YOUR CODE HERE
[m, n, c] = size(I);
I_downsampled_notpre = zeros(n/2, m/2, c);
for i = 1:3 
    temp = I(:, :, i);
    temp = downsample(temp, 2);
    temp = downsample(temp', 2);
    I_downsampled_notpre(:, :, i) = temp;
end
I_rec_notpre = zeros(m, n, c);
for i = 1:3 
    temp = I_downsampled_notpre(:, :, i);
    temp = upsample(temp, 2);
    temp = upsample(temp', 2);
    I_rec_notpre(:, :, i) = temp;
    I_rec_notpre(:, :, i)=  4*prefilterlowpass2d(I_rec_notpre(:, :, i), kernel);
end
% Evaluation without prefiltering
% I_rec_notpre is the reconstructed image WITHOUT prefiltering
PSNR_notpre = calcPSNR(I, I_rec_notpre);
fprintf('Reconstructed image, not prefiltered, PSNR = %.2f dB\n', PSNR_notpre)

% with prefiltering
% YOUR CODE HERE
I_downsampled_notpre = zeros(n/2, m/2, c);
%Downsample
for i = 1:3 
    temp = prefilterlowpass2d(I(:, :, i), kernel);
    temp = downsample(temp, 2);
    temp = downsample(temp', 2);
    I_downsampled_notpre(:, :, i) = temp;
end
I_rec_pre = zeros(m, n, c);
%Reconstruct
for i = 1:3 
    temp = I_downsampled_notpre(:, :, i);
    temp = upsample(temp, 2);
    temp = upsample(temp', 2);
    I_rec_pre(:, :, i) = temp;
    I_rec_pre(:, :, i)=4*prefilterlowpass2d(I_rec_pre(:, :, i), kernel);
end
% Evaluation with prefiltering
% I_rec_pre is the reconstructed image WITH prefiltering
PSNR_pre = calcPSNR(I, I_rec_pre);
fprintf('Reconstructed image, prefiltered, PSNR = %.2f dB\n', PSNR_pre)


%Subtask E1_2c
I_pre = zeros(m, n, c);
for i = 1:3 
    I_pre(:, :, i) = prefilterlowpass2d(I(:, :, i), kernel);
end
figure;
subplot(1, 3, 1);
imshow(uint8(I)), title('W/O filtering');
subplot(1, 3, 2);
imshow(uint8(I_pre)), title('With filtering');
subplot(1, 3, 3);
imshow(abs(uint8(I) - uint8(I_pre))), title('Difference image');













% put all the sub-functions called in your script here
function pic_pre = prefilterlowpass2d(picture, kernel)
% YOUR CODE HERE
    pic_pre = conv2(picture, kernel, 'same');
end

function MSE = calcMSE(Image, recImage)
% YOUR CODE HERE
    [m, n, c] = size(Image);
    Image = double(Image);
    recImage = double(recImage);
    MSE = 1/(m * n * c) * sum((Image - recImage).^2, 'all');
end

function PSNR = calcPSNR(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
%
% Output        : PSNR     (Peak Signal to Noise Ratio)
% YOUR CODE HERE
% call calcMSE to calculate MSE
    PSNR=10*log10((2^8-1).^2/calcMSE(Image, recImage));
end