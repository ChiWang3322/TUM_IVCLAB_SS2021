function image = InvVectorQuantizer(qImage, clusters, block_size)
%  Function Name : VectorQuantizer.m
%  Input         : qImage     (Quantized Image)
%                  clusters   (Quantization clusters)
%                  block_size (Block Size)
%  Output        : image      (Dequantized Images)
    [Mq, Nq, Cq] = size(qImage);
    image = zeros(Mq * block_size, Nq * block_size, Cq);
    for c = 1 : Cq
        for m = 1 : Mq
            for n = 1 : Nq
                index = qImage(m, n, c);
                image( (m - 1) * block_size + 1 : m * block_size,...,
                         (n - 1) * block_size + 1 : n * block_size, c) = reshape(clusters(index, :), [block_size, block_size]);
            end
        end
    end
end