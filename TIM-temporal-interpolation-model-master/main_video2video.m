
% ���ڶ�ԭ��Ƶ����temporal interpolation model

clear all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nf_output = 16; % number of output frames
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = 'EP02_01f.avi';
obj = VideoReader(filename);
numframe = obj.NumberOfFrames;
length = obj.Width * obj.Height;
fprintf('frames number is %d֡\n',numframe);
img_vec = zeros(length, numframe);

for i = 1:numframe
    frame = read(obj,i);
    gray = rgb2gray(frame);
    img_vec(:,i) = gray(:);
end

model = TrainPGM(img_vec);
% pos = 0.5*1/numframe:0.5*1/numframe:1;
pos = 0:1.0/(nf_output-1):1
Y = synPGM(model, pos);
Y = uint8(Y);
videoName = './test.avi';
startFrame = 1;
%endFrame = 2 * numframe;
endFrame = nf_output;
aviobj=VideoWriter(videoName);  %����һ��avi��Ƶ�ļ����󣬿�ʼʱ��Ϊ��
aviobj.FrameRate=obj.FrameRate;
open(aviobj);
for i = startFrame:endFrame
    frames = reshape(Y(:,i), obj.Height, obj.Width);
    writeVideo(aviobj, frames);
end
close(aviobj)

    
