clc;
clear all;

root = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticflow/';
strainimg_output =  '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticalstrain_image/';
output_format = 'jpg';
expressions = dir(root);
expressions = expressions(3:end);
num_express = size(expressions, 1);
for i = 1:num_express
    opticflows_paths = dir([root, expressions(i).name, '/']);
    opticflows_paths = opticflows_paths(3:end);
    opticflows_paths = sortObj(opticflows_paths);
    for j = 1:size(opticflows_paths, 1)
        opticflows_path = [root, expressions(i).name, '/', opticflows_paths(j).name, '/'];
        output_path = [strainimg_output, expressions(i).name, '/', opticflows_paths(j).name, '/'];
        opticflows = dir(opticflows_path);
        opticflows = opticflows(3:end);
        opticflows = sortObj(opticflows);
        for k = 1:size(opticflows, 1)
            opticflow = [opticflows_path, opticflows(k).name];
            [optical_strain, ~, ~, ~] = zxl_compute_opticalstrain(opticflow);
            if(~exist(output_path, 'file'))
                mkdir(output_path); 
            end
            imwrite(optical_strain, [output_path, num2str(k), '.', output_format]);
        end
        disp([int2str(i), '/', int2str(j)]);
    end
end