function H = stats_cond(image)
%  Input         : image (Original Image)
%
%  Output        : H   (Conditional Entropy)
    jpmf = zeros(256, 256);
    [M, N, C] = size(image);
    for c=1 : C
        for m=1 : M
            for n=2 : N
                jpmf(image(m, n, c), image(m,n - 1,c)) =  ...,
                jpmf(image(m, n, c), image(m,n - 1,c)) + 1;
            end
        end
    end
    
    jpmf = jpmf / sum(jpmf(:));
    
    range = 0:255;
    mpmf = hist(image(:), range);
    mpmf = mpmf / sum(mpmf(:));
    cpmf = jpmf ./ mpmf;
    cpmf( isnan(cpmf) ) = 0;
    cpmf( isinf(cpmf) ) = 0;
    
    temp = jpmf.*log2(cpmf);
    temp( isinf(temp) ) = 0;
    temp( isnan(temp) ) = 0;
    H=-sum(temp(:));
end

