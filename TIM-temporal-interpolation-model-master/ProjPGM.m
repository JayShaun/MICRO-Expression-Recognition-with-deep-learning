function Y = ProjPGM(X,model)
% Y = ProjPathGraphModel(X,model)
%
% Function that projects data points into the low-dimensional space within 
% which the training data are projected onto the curve embedded in the path
% graph.
% input:
%        X - an array of column vectors that contains the input feature
%        vectors
%        model - model trained by function TrainPGM
% output:
%        Y - projected points in the low-dim manifold (column-wise)
%
% Reference
%   Z. Zhou, G. Zhao and M. Pietikainen. Towards a practical lipreading
%   system. CVPR'11, pp.137-144, 2011. 
%
% Please refer the above reference if you use this code in your work. 

if strcmpi(class(X),'double')~=1
    X = double(X);
end

Y = diag(1./model.m)*model.W'*model.U'*(X-repmat(model.mu,1,size(X,2)));