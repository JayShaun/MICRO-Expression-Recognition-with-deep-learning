function X = synPGM(model,pos)
% X = synPGM(model,pos)
%
% Project points on the embedded curve back into the original high-dim 
% space
%
% Input: 
%        model - model trained by function TrainPGM
%        pos - array containing positions of the positions on the curve
%        ([0,1])
% Ouput:
%        X - column-wise matrix containing the synthesized vectors
%
% Reference
%   Z. Zhou, G. Zhao and M. Pietikainen. Towards a practical lipreading
%   system. CVPR'11, pp.137-144, 2011. 
%
% Please refer the above reference if you use this code in your work.

n = model.n;

% Rescale positions in [1/n,1]
pos = pos*(1-1/n)+1/n;

% Synthesis
X = zeros(size(model.U,1),length(pos));
ndim = size(model.W,1);
for i = 1 : length(pos)
    v = zeros(ndim,1);
    for k = 1 : ndim
        v(k) = sin(pos(i)*k*pi+pi*(n-k)/(2*n));
    end
    X(:,i)=model.U*(model.W'\(v.*model.m))+model.mu;
end