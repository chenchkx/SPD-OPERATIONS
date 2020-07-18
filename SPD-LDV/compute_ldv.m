%%  computing the Local Difference Vectors(LDV) on SPD manifold. (https://github.com/Kai-Xuan/MyNote/tree/master/ML/SPD-LDV)
% Written by Kai-Xuan Chen (e-mail: kaixuan_chen_jsh@163.com).If you find any bug, please contact me.
%   
% If you find this code useful for your research, maybe you can cite the following paper:
%{
    @article{chen2020covariance,
      title={Covariance Descriptors on a Gaussian Manifold and their Application to Image Set Classification},
      author={Chen, Kai-Xuan and Ren, Jie-Yi and Wu, Xiao-Jun and Kittler, Josef},
      journal={Pattern Recognition},
      pages={107463},
      year={2020},
      publisher={Elsevier}
    }
%} 
%   input: 
%         X : an SPD matrix 
%         Y : an SPD matrix
%         type : the type of metrices on the SPD manifold.
%   output:
%         rie_ldv : Riemannian LDV on the SPD manifold
% 
function rie_ldv = compute_ldv(X,Y,type)

    X_m0d5 = X^(-0.5); 
    switch type
        case 'A'    % Affine Invariant Riemannian Metric 
            X_0d5 = X^(0.5);   
            temp_norm = logm(X_m0d5*Y*X_m0d5);
            temp_gradient = 2*X_0d5*temp_norm*X_0d5;
%             temp_dis = compute_distance(X,Y,type); 
            temp_dis = norm(temp_norm,'fro'); % faster here 
            
        case 'S'    % Stein divergence 
            temp_gradient = X*((X+Y)^-1)*X - 0.5*X;
            temp_dis = compute_distance(X,Y,type); 
          
        case 'J'    % Jeffrey divergence 
            X_m1 = X^(-1);  Y_m1 = Y^(-1);
            temp_gradient = 0.5*X*(Y_m1 - X_m1*Y*X_m1)*X;
%             temp_dis = compute_distance(X,Y,type); 
            temp_dis = sqrt(0.5*trace(X_m1*Y)+0.5*trace(Y_m1*X) - size(X,1)); % faster here       
            
        case 'L'    % Log-Euclidean Metric 
            temp_norm = (logm(X)-logm(Y));
            temp_gradient = X^(-1)*temp_norm;
            temp_gradient = 2*X*(temp_gradient + temp_gradient')*X;
%             temp_dis = compute_distance(X,Y,type); 
            temp_dis = norm(temp_norm,'fro'); % faster here 
            
    end
        
    current_gradient = X_m0d5*temp_gradient*X_m0d5 ;    % normalization. 
    current_dis = temp_dis;
    if current_dis < 1e-10 || norm(current_gradient,'fro') < 1e-10
        temp_ldv = zeros(size(current_Gradient));
    else
        temp_ldv = (current_gradient/norm(current_gradient,'fro'))*current_dis;   
    end   
    rie_ldv = map2IDS_vectorize(temp_ldv, 0);
    
end


%% Code from: https://github.com/Kai-Xuan/MyNote/tree/master/ML/SPD-Metrics
function out_dis = compute_distance(X,Y,type)
    switch type
        case 'A'    % Affine Invariant Riemannian Metric 
            tmpEig =  eig(X,Y);  
            tmp_dis = sum(log(tmpEig).^2);            
        case 'S'    % Stein divergence 
            t = log(det(0.5*(X+Y))) - 0.5* (log(det(X)) + log(det(Y)));
            if t == inf || isnan(t)
                eigX = eig(X);
                eigX (eigX <= 0) = eps;
                eigY = eig(Y);
                eigY (eigY <= 0) = eps;
                t = real(sum(log(eig(0.5*(X+Y))))) - 0.5 *(real(sum(log(eigX))) + real(sum(log(eigY))));    
            end
            tmp_dis = t;      
        case 'J'    % Jeffrey divergence 
            tmp_dis = 0.5*trace(inv(X)*Y)+0.5*trace(inv(Y)*X) - size(X,1);            
        case 'L'    % Log-Euclidean Metric 
            tmp_dis = norm((logm(X)-logm(Y)),'fro').^2;
    end
    if tmp_dis <= 1e-15
        out_dis = 0;
    else
        out_dis = sqrt(tmp_dis);
    end
    
end

%% Code from: https://github.com/mfaraki/Riemannian_VLAD
function y = map2IDS_vectorize(inMat, map2IDS)
    if map2IDS == 1
        inMat = logm(inMat);    
    end
    offDiagonals = tril(inMat,-1) * sqrt(2);
    diagonals = diag(diag(inMat));
    vecInMat = diagonals + offDiagonals; 
    vecInds = tril(ones(size(inMat)));
    map2ITS = vecInMat(:);
    vecInds = vecInds(:);
    y = map2ITS(vecInds == 1) ;
end


