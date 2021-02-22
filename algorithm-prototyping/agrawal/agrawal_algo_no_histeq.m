a = imread('C:\Users\Noelle\Documents\school\iqp\AI repo\histogram_equalization\algorithm-prototyping\images\test1.png');
a = rgb2gray(a); %converts to grayscale

r = size(a,1);
c = size(a,2);
ah = uint8(zeros(r,c));
n = r*c;
f = zeros(256,1);
pdf = zeros(256,1);
cdf = zeros(256,1);
cum = zeros(256,1);
out = zeros(256,1);

for i=1:r
for j=1:c

value = a(i,j);
f(value+1) = f(value+1)+1;
pdf(value+1) = f(value+1)/n;
end
end

sum = 0;
L = 255;

for i=1:size(pdf)
sum = sum+f(i);
cum(i) = sum;
cdf(i) = cum(i)/n;
out(i) = round(cdf(i)*L);
end

for i=1:r
for j=1:c
ah(i,j)=out(a(i,j)+1);
end
end

figure,imshow(ah);
title('agrawal: without histeq')
he = histeq(a);
figure, imshow(he);
title('agrawal: with histeq')