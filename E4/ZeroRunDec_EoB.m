function dst = ZeroRunDec_EoB(src, EoB)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)
    [M, N] = size(src);
    dst = zeros(1, 100 * N);
    last_el_is_zero = 0;
    pointer = 1;
    for i = 1:length(src)
        temp = src(i);
        if src(i) == EoB
            if mod(pointer, 64) == 0    %at the last position of one block
                dst(pointer) = 0;
                pointer = pointer + 1;
            else
                num_zeros = 64 - mod(pointer, 64) + 1;
                dst(pointer : pointer + num_zeros - 1) = zeros(1, num_zeros);
                pointer = pointer + num_zeros;
            end
        %Last symbol is 0 and the current symbol is not 0
        elseif (src(i) == 0) && (~last_el_is_zero)
            last_el_is_zero = 1;
            dst(pointer) = 0;
            pointer = pointer + 1;
        elseif last_el_is_zero
            dst(pointer:pointer + src(i) - 1) = zeros(1, src(i));
            last_el_is_zero = 0;
            pointer = pointer + src(i);
        else
            dst(pointer) = src(i);
            pointer = pointer + 1;
        end
        
    end
    %Process the end of dst
    dst(pointer : end) = [];
end