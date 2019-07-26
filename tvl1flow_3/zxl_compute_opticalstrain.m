function [optical_strain, e_xy, e_xx, e_yy] = zxl_compute_opticalstrain(opticalflow_file)
% clc;
% clear all;
opticalflow = readFlowFile(opticalflow_file);


%% 计算optical strain
% 参考：1)https://www.sciencedirect.com/science/article/pii/S0923596517302436；
        %论文Macro- and micro-expression spotting in long videos using spatio-temporal strain
% 参考：2)output_strain.m (作者放出的代码并没有给出计算optical strain的代码(os.m)，故在这里自己实现)
% 另外，作者本来应该是按照参考1）把magnitude，orientation和normal strain全都计算了，当成图像的三个channel，但后来只
% 用了作者论文里提到的optical strain作为1个channel的图像，可能是因为三个全用效果不好.可以自己试试，这里只实现optical strain。

of_x = opticalflow(:, :, 1);
of_y = opticalflow(:, :, 2);
dx = 2;
dy = 2;
[m, n] = size(of_x);
e_xx = zeros(m, n);
e_yy = zeros(m, n);
e_x_y = zeros(m, n);
e_y_x = zeros(m, n);

for i = 1 : n
   if i <= dx
       e_xx(:, i) = (of_x(:, i+dx) - of_x(:, i)) / dx;
   elseif i > n-dx
       e_xx(:, i) = (of_x(:, i) - of_x(:, i-dx)) / dx;
   else
       e_xx(:, i) = (of_x(:, i+dx) - of_x(:, i-dx)) / (2*dx);
   end   
end

for i = 1 : m
   if i <= dy
       e_yy(i, :) = (of_y(i+dy, :) - of_y(i, :)) / dy;
   elseif i > n-dx
       e_yy(i, :) = (of_y(i, :) - of_y(i-dy, :)) / dy;
   else
       e_yy(i, :) = (of_y(i+dy, :) - of_y(i-dy, :)) / (2*dy);
   end   
end

for i = 1 : m
   if i <= dy
       e_x_y(i, :) = (of_x(i+dy, :) - of_x(i, :)) / dy;
   elseif i > n-dx
       e_x_y(i, :) = (of_x(i, :) - of_x(i-dy, :)) / dy;
   else
       e_x_y(i, :) = (of_x(i+dy, :) - of_x(i-dy, :)) / (2*dy);
   end   
end

for i = 1 : n
   if i <= dx
       e_y_x(:, i) = (of_y(:, i+dx) - of_y(:, i)) / dx;
   elseif i > n-dx
       e_y_x(:, i) = (of_y(:, i) - of_y(:, i-dx)) / dx;
   else
       e_y_x(:, i) = (of_y(:, i+dx) - of_y(:, i-dx)) / (2*dx);
   end   
end
e_xy = 0.5 * (e_x_y + e_y_x);

os_magnitude = sqrt(e_xx.^2 + e_yy.^2 + 2 * e_xy.^2);
% 下面做了normalize处理，目的是为了以图像输出
os_magnitude = 255 ./ (max(max(os_magnitude))-min(min(os_magnitude))) .* (os_magnitude - min(min(os_magnitude)));
os_magnitude = uint8(os_magnitude);
optical_strain = os_magnitude;
%imshow(os_magnitude)