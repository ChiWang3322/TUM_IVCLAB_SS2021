function image = InvLloydMax(qImage, clusters)
%  Input         : qImage   (Quantized Image)
%                  clusters (Quantization Table)
%  Output        : image    (Recovered Image)
    image = zeros(size(qImage));
    for i = 1:length(clusters)
        image(qImage == i) = clusters(i);
    end
end