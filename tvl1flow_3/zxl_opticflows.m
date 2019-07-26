function zxl_opticflows(frames_path, output_path)

if(~exist(output_path, 'file'))
    mkdir(output_path); 
end

frames = dir(frames_path);
frames = frames(3:end);
frames = sortObj(frames);
num_frames = size(frames, 1);
compare_source_idx =1;
compare_target_idx = 2;
flow_idx = 1;

for i = 1: num_frames
    system_str = './tvl1flow ';
    compare_source = [frames_path, frames(compare_source_idx).name];%这里不以i和i+1作为索引的目的在于：希望程序能够计算任意两帧之间的光流
    compare_target = [frames_path, frames(compare_target_idx).name];%另外，也是为了后面保证输出光流数量与输入帧的数量相等，即把这些帧认为是循环的
    
    system_str = [system_str, compare_source, ' ', compare_target, ' ', [output_path, num2str(flow_idx), '.flo']];
    %system_str = [system_str];
    %system("chmod +x tvl1flow")
    
    [~, ~] = system(system_str);

    compare_source_idx = compare_source_idx + 1;
    compare_target_idx = compare_target_idx + 1;
    if compare_target_idx > num_frames
        compare_target_idx = 1;
    end
    flow_idx = flow_idx + 1;
    disp([int2str(i), '/', num_frames]);
end