
% ���ڶ�ԭ��Ƶ����temporal interpolation model

clear all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nf_output = 16; % number of output frames
output_path = './output_tim/'; 
output_format = 'jpg';

if(~exist(output_path, 'file'))
    mkdir(output_path); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

frames_path = '/media/zxl/other/pjh/datasetsss/CK+/cropped/anger/1/';
frames = dir(frames_path);
frames = frames(3:end); %remove . and ..
frames = sortObj(frames);
aframe = imread([frames_path, frames(1).name]);

[width, height, channel] = size(aframe);
length = width * height;
numframe = size(frames, 1);
img_vec = zeros(length, numframe);

for i = 1:numframe
    frame = imread([frames_path, frames(i).name]);
    if channel > 1
        frame = rgb2gray(frame);
    end
    img_vec(:,i) = frame(:);
end

model = TrainPGM(img_vec);
pos = 0:1.0/(nf_output-1):1;
Y = synPGM(model, pos);
Y = uint8(Y);
startFrame = 1;
endFrame = nf_output;

for i = startFrame:endFrame
    frames = reshape(Y(:,i), width, height);
    imwrite(frames, [output_path, num2str(i), '.', output_format]);
end
