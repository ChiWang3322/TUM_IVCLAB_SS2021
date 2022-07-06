function image = InvUniQuant(qImage, bits)
%  Input         : qImage (Quantized Image)
%                : bits (bits available for representatives)
%
%  Output        : image (Mid-rise de-quantized Image)
qImage=qImage+0.5;
image = floor(256 / (2 ^ bits) * qImage);
end