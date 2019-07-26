function zxl_opticflows2imgs(opticflows_path, output_path)

if(~exist(output_path, 'file'))
    mkdir(output_path); 
end

output_format = 'jpg';
opticflows = dir(opticflows_path);
opticflows = opticflows(3:end);
opticflows = sortObj(opticflows);
num_opticflows = size(opticflows, 1);


for i = 1: num_opticflows
    opticflow = readFlowFile([opticflows_path, opticflows(i).name]);
    of_x = opticflow(:, :, 1);
    of_y = opticflow(:, :, 2);

    scaling = 16;
    shifting = 128;
    mag = sqrt(of_x.^2 + of_y.^2) * scaling + shifting;  % Euclidean Distance
    mag = uint8(min(mag, 255));

    of_x = of_x * scaling + shifting;
    of_y = of_y * scaling + shifting;
    of_x = min(of_x,255);
    of_x = max(of_x , 0);
    of_y = min(of_y, 255);
    of_y = max(of_y, 0);
    of_x = uint8(of_x);
    of_y = uint8(of_y);

    flow_image(:, :, 1) = of_x;
    flow_image(:, :, 2) = of_y;
    flow_image(:, :, 3) = mag;
    
    imwrite(flow_image, [output_path, num2str(i), '.', output_format]);
end