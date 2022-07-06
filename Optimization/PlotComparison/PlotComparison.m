figure;
load still_codec_values
plot(BPP_mean_still, PSNR_mean_still, '--o', 'LineWidth' , 1.5, 'MarkerSize', 6);

grid on
hold on

load still_codec_values_optimized.mat
plot(BPP_mean_still, PSNR_mean_still, '--o', 'LineWidth' , 1.5, 'MarkerSize', 6);

load video_codec_values.mat
plot(BPP_mean_video, PSNR_mean_video, '--o', 'LineWidth' , 1.5, 'MarkerSize', 6);

load video_codec_values_optimized.mat
plot(BPP_mean_video, PSNR_mean_video, '--o', 'LineWidth' , 1.5, 'MarkerSize', 6);


legend('Chapter4 curve', 'Optimized chapter4 curve(adaptive huffmann coding)', 'Chapter5 Curve', 'Optimized chapter5 curve', 'Location', 'southeast');
title('Rate-Distortion Plot');
xlim([0.2 4]);
xlabel('Bit Rate [bits/pixel]');
ylabel('PSNR [dB]');