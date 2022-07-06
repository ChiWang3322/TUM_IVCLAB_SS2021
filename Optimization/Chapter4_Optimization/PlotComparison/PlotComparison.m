figure;
load still_codec_values
plot(BPP_mean_still, PSNR_mean_still, '--o', 'LineWidth' , 1.5, 'MarkerSize', 6);

grid on
hold on

load still_codec_values_adaptive_huff.mat
plot(BPP_mean_still, PSNR_mean_still, '--o', 'LineWidth' , 1.5, 'MarkerSize', 6);

load video_codec_values.mat
plot(BPP_mean_video, PSNR_mean_video, '--o', 'LineWidth' , 1.5, 'MarkerSize', 6);

load video_codec_values_SSD_mul_optim_huff.mat
plot(BPP_mean_video, PSNR_mean_video, '--o', 'LineWidth' , 1.5, 'MarkerSize', 6);


legend('Non-optimized Still image codec', 'Optimized Still image codec', 'Non-optimized Video image codec', 'Optimized Video image codec', 'Location', 'southeast');
title('Rate-Distortion Plot');
xlabel('Bit Rate [bits/pixel]');
ylabel('PSNR [dB]');