%For rgb to gray compression
R = 8;
offset = [0.5, 0.2];
PSNR = 16.62;
figure;
hold on;
plot(R, PSNR, 'bo');
plot(R, 15.46, 'bo');
plot(R, 17.02, 'bo');
text(R + offset(1), PSNR, 'smandril\_rgb2gray');
text(R + offset(1), 15.46, 'lena\_rgb2gray');
text(R + offset(1), 17.02, 'monarch\_rgb2gray');
%For lowpass filter
R = 6;
order = ['lena', 'mandril', 'monarch'];
PSNR_notpre = [30.51, 27.62, 30.72];
PSNR_pre = [29.87, 26.86, 29.44];
plot(R, PSNR_pre(1), 'rx');
plot(R, PSNR_pre(2), 'rx');
plot(R, PSNR_pre(3), 'rx');
text(R + offset(1), PSNR_pre(1), 'lena\_lowpass\_pre');
text(R + offset(1), PSNR_pre(2), 'smandril\_lowpass\_pre');
text(R + offset(1), PSNR_pre(3), 'monarch\_lowpass\_pre');
%For FIR filter
R = 6;
order = ['lena', 'mandril', 'monarch'];
PSNR_pre = [31.34, 23.71, 32.23];
plot(R, PSNR_pre(1), 'g*');
plot(R, PSNR_pre(2), 'g*');
plot(R, PSNR_pre(3), 'g*');
text(R + offset(1), PSNR_pre(1), 'lena\_FIR\_pre');
text(R + offset(1), PSNR_pre(2), 'smandril\_FIR\_pre');
text(R + offset(1), PSNR_pre(3), 'monarch\_FIR\_pre');
%For Chroma subsampling
R = 12;
order = ['lena', 'mandril', 'monarch'];
PSNR_pre = [38.643, 30.07, 49.48];
plot(R, PSNR_pre(1), 'kd');
plot(R, PSNR_pre(2), 'kd');
plot(R, PSNR_pre(3), 'kd');
text(R + offset(1), PSNR_pre(1), 'lena\_chroma\_subsampling');
text(R + offset(1), PSNR_pre(2), 'smandril\_chroma\_subsampling');
text(R + offset(1), PSNR_pre(3), 'monarch\_chroma\_subsampling');

%Add title and label
hold off;
title('R-D plot');
xlim([0, 25]);;
xlabel('Bitrate(in bits/pel)');
ylabel('PSNR(in dB)');