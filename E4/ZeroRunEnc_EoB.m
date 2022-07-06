function zze = ZeroRunEnc_EoB(zz, EOB)
%  Input         : zz (Zig-zag scanned sequence, 1xN)
%                  EOB (End Of Block symbol, scalar)
%
%  Output        : zze (zero-run-level encoded sequence, 1xM)
    zze = zeros(size(zz));  %pre-allocate memory
    pointer_zze = 1;    %Using indexing, which is much faster
    len = length(zz);
    zeros_rep = 0;  %Number of repetitions of zeros
    for i = 1:len
        temp = zz(i);
        % When current symbol is 0
        if temp == 0
            % When this 0 is the first 0 in a string
            if zeros_rep == 0
                zze(pointer_zze:pointer_zze + 1) = [0, 0];
                zeros_rep = 1;
                pointer_zze = pointer_zze + 2;
            % When this zeros is the following 0 in a string, rep += 1
            else
                zze(pointer_zze - 1) = zeros_rep;
                zeros_rep = zeros_rep + 1;
            end
        % When the current symbol is not 0
        else 
            zeros_rep = 0;
            zze(pointer_zze) = temp;
            pointer_zze = pointer_zze + 1;
        end
        % Check if the last symbol of 8x8 block is 0
        if (mod(i, 64) == 0) && (zz(i) == 0)
            zze(pointer_zze - 2) = EOB;
            pointer_zze = pointer_zze - 1;
            zeros_rep = 0;
        end
    end
    
    zze(pointer_zze:end) = [];
    % Process end of the sequence
    if zze(end) == 0 | zze(end - 1) == 0
        zze(end - 1:end) = [];
        zze(end + 1) = EOB;
    end
end


    