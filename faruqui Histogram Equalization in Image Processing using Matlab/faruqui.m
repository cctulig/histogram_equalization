i = imread('C:\Users\Noelle\Documents\school\iqp\AI repo\histogram_equalization\algorithm-prototyping\images\test1.png');
ih = histeq(i);

subplot(2,2,1), imshow(i), title('Original Image');
subplot(2,2,2), imshow(ih), title('Histogram Equalized Image');
subplot(2,2,3), imhist(i), title('Histogram of Original Image');
subplot(2,2,4), imhist(ih), title('Histogram of Equalized Image');