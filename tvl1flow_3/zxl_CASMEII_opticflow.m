clc;
clear all;

root = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM/';
flow_output =  '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticflow/';

expressions = dir(root);
expressions = expressions(3:end);
num_express = size(expressions, 1);
for i = 1:num_express
    frames_paths = dir([root, expressions(i).name, '/']);
    frames_paths = frames_paths(3:end);
    frames_paths = sortObj(frames_paths);
    for j = 1:size(frames_paths, 1)
        frame_path = [root, expressions(i).name, '/', frames_paths(j).name, '/'];
        output_path = [flow_output, expressions(i).name, '/', frames_paths(j).name, '/'];
        zxl_opticflows(frame_path, output_path);
    end
end


