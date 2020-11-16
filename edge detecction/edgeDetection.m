%Detect Edges in Images
%This example shows how to detect edges in an image using both the Canny edge detector and the Sobel edge detector.
%Read image and display it.
I = imread('C:\Users\Noelle\Documents\school\iqp\AI repo\histogram_equalization\algorithm-prototyping\images\test3.png');
I = rgb2gray(I); %won't work on color image because there are 3 color channels
windowWidth = 10; % Whatever you want.  More blur for larger numbers.
kernel = ones(windowWidth) / windowWidth ^ 2;
I = imfilter(I, kernel); % Blur the image.

%I = wiener2(I,[5 5]);%experiments with removing skin texture
imshow(I)

%Apply both the Sobel and Canny edge detectors to the image and display them for comparison.
BW1 = edge(I,'sobel');
BW2 = edge(I,'canny');
figure;
imshowpair(BW1,BW2,'montage')
title('Sobel Filter and Canny Filter');
%Copyright 2015 The MathWorks, Inc.
