kernel = [0, 0, 3, 2, 2, 2, 3, 0, 0;
                   0, 2, 3, 5, 5, 5, 3, 2, 0;
                   3, 3, 5, 3, 0, 3, 5, 3, 3;
                   2, 5, 3, -12, -23, -12, 3, 5, 2;
                   2, 5, 0, -23, -40, -23, 0, 5, 2;
                   2, 5, 3, -12, -23, -12, 3, 5, 2;
                   3, 3, 5, 3, 0, 3, 5, 3, 3;
                   0, 2, 3, 5, 5, 5, 3, 2, 0;
                   0, 0, 3, 2, 2, 2, 3, 0, 0];

%disp(kernel)

image = imread("pepper.ascii.pgm");
image = im2gray(image);

filterResult = conv2(kernel,image);

imshow(filterResult);