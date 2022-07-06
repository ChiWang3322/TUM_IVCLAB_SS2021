function pic_pre = prefilterlowpass2d(picture, kernel)
% YOUR CODE HERE
    pic_pre = conv2(picture, kernel, 'same');
end