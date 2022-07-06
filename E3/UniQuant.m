function qImage = UniQuant(image, bits)
%  Input         : image (Original Image)
%                : bits (bits available for representatives)
%
%  Output        : qImage (Quantized Image)
image = image / 256;
qImage = floor( image * (2^bits));
end