clear all;
clc;
%%
% TIM插值程序, nf_output表示插值输出的序列长度
%%
nf_output = 10;
root = '/media/zxl/other/pjh/datasetsss/CASME_II/';
output_root = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM/';
% expressions = ['disgust', 'fear', 'happiness', 'others', 'repression', 'sadness', 'surprise'];
expressions = dir(root);
expressions = expressions(3:end);
num_express = size(expressions, 1);
for i = 1:num_express
    frames_paths = dir([root, expressions(i).name, '/']);
    frames_paths = frames_paths(3:end);
    frames_paths = sortObj(frames_paths);
    for j = 1:size(frames_paths, 1)
        frame_path = [root, expressions(i).name, '/', frames_paths(j).name, '/'];
        output_path = [output_root, expressions(i).name, '/', frames_paths(j).name, '/'];
        TIM_img2img(nf_output, frame_path, output_path);
    end
end

