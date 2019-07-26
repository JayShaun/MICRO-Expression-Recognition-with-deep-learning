function model = TrainPGM(X)
% model = TrainPGM(X)
%
% Train a linear model that projects a sequence of high-dim vectors onto
% a deterministic low-dim curve embedded in a path graph. The feature
% vectors are assumed to be linearly independent.
%
% input:
%        X - an array of column vectors that contains the input feature
%        vectors
% output:
%        model - trained model
%
% Reference
%   Z. Zhou, G. Zhao and M. Pietikainen. Towards a practical lipreading
%   system. CVPR'11, pp.137-144, 2011. 
%
% Please refer the above reference if you use this code in your work. 

    X = double(X);
    [~,N] = size(X);

    % check linear independency
    if rank(X)<N 
        error('TrainPGM: Invalid input.');
    end

    mu = mean(X,2);    % 平均特征向量
    X = X - repmat(mu,1,N);
    [U,S,V] = svd(X,'econ');
    S(N,:) = [];
    S(:,N) = [];
    V(:,N) = [];
    U(:,N) = [];
    Q = S*V';
    % clear S;
    % clear V;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Graph Embedding
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    G = getPathWeightMatrix(N);
    L = getLapacian(G);

    [V0,~] = eig(L);
    V0(:,1) = [];
    W = (Q*Q')\(Q*V0);   % Q'Vk = yk,解出Vk

    m = zeros(N-1,1);
    for j = 1 : N-1
        m(j) = Q(:,1)'*W(:,j)/sin(1/N*j*pi+pi*(N-j)/(2*N));
    end

    model.W = W;
    model.U = U;
    model.mu = mu;
    model.n = N;
    model.m = m;
end

function G = getPathWeightMatrix(N)
    G = zeros(N,N);
    for i= 1:N-1
        G(i,i+1) = 1;
        G(i+1,i) = 1;
    end
end

function L = getLapacian(G)
    N = size(G,1);

    D = zeros(N,N);
    D((0:N-1)*N+(1:N)) = sum(G);
    L = D - G;
end