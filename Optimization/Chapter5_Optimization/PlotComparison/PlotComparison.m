figure;
load video_codec_values
plot(BPP_mean_video, PSNR_mean_video, '--o', 'LineWidth' , 2, 'MarkerSize', 8);
% for i = 1 : length(scales_still)
%     text(BPP_mean_still(i) + 0.1, PSNR_mean_still(i), num2str(scales_still(i), '%.2f'));
% end

grid on
hold on


load video_codec_values_SSD_mul_optim_huff.mat
plot(BPP_mean_video, PSNR_mean_video, '--o', 'LineWidth' , 2, 'MarkerSize', 8);

legend('Non-optimized Video codec', 'Optimized video codec', 'Location', 'southeast');
title('Rate-Distortion Plot');
xlabel('Bit Rate [bits/pixel]');
ylabel('PSNR [dB]');