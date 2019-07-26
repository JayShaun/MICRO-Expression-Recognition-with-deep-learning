clc;
clear all;

root = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticflow/';
flowimg_output =  '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticflow_image/';

expressions = dir(root);
expressions = expressions(3:end);
num_express = size(expressions, 1);
for i = 1:num_express
    opticflows_paths = dir([root, expressions(i).name, '/']);
    opticflows_paths = opticflows_paths(3:end);
    opticflows_paths = sortObj(opticflows_paths);
    for j = 1:size(opticflows_paths, 1)
        opticflows_path = [root, expressions(i).name, '/', opticflows_paths(j).name, '/'];
        output_path = [flowimg_output, expressions(i).name, '/', opticflows_paths(j).name, '/'];
        zxl_opticflows2imgs(opticflows_path, output_path);
        disp([int2str(i), '/', int2str(j)]);
    end
end