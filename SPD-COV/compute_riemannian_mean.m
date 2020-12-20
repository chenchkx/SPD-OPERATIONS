%% Riemannian means on the Symmetric Positive Definite(SPD) manifold. (https://github.com/Kai-Xuan/MyNote/tree/master/ML/SPD-Means)
% Points are SPD matrices: spd_matrices(:,:,1), ..., spd_matrices(:,:,N)
% Rewritten by Kai-Xuan Chen (e-mail: kaixuan_chen_jsh@163.com),If you find any bugs, please contact me.
% you also can refer to: https://github.com/mfaraki/Riemannian_VLAD & https://github.com/oryair/ParallelTransportSPDManifold
% 
% If you find this code useful for your research, we appreciate it very much if you can cite our related works:
% @article{chen2020covariance,
%   title={Covariance Descriptors on a Gaussian Manifold and their Application to Image Set Classification},
%   author={Chen, Kai-Xuan and Ren, Jie-Yi and Wu, Xiao-Jun and Kittler, Josef},
%   journal={Pattern Recognition},
%   pages={107463},
%   year={2020},
%   publisher={Elsevier}
% }
% 
% input
%     spd_matrices : spd_matrices(:,:,1), ..., spd_matrices(:,:,N) are N SPD matrix
%     type_metric: Riemannian metric used for SPD manifold('A':AIRM, 'S':Stein, 'J':Jeffrey, 'L':LEM. )
% output
%     mean_center: the Riemannian center computed by this function

function [mean_center] = compute_riemannian_mean(spd_matrices,type_metric,max_iter)
    
    [dims,~,num_spd] = size(spd_matrices);
    M  = mean(spd_matrices, 3);
    if (nargin < 3)
        max_iter = 15;
    end
    switch type_metric

        case 'A'    % computing mean via AIRM       
            for ite_th = 1 : max_iter
                A = M ^ (1/2);      %-- A = C^(1/2)
                B = A ^ (-1);       %-- B = C^(-1/2)

                S = zeros(size(M));
                for j_th = 1 : num_spd
                    C = spd_matrices(:,:,j_th);
                    S = S + A * logm(B * C * B) * A;
                end
                S = S / num_spd;

                M = A * expm(B * S * B) * A; 

                eps = norm(S, 'fro');
                if (eps < 1e-6)
                    break;
                end
            end
            
        case 'S'    % computing mean via Stein
            for ite_th = 1:max_iter
                tmpX = zeros(dims);
                for j_th = 1:num_spd
                    tmpX = real(tmpX + inv((spd_matrices(:,:,j_th) + M)/2));
                end
                tmpX = tmpX/num_spd;
                M = inv(tmpX);
            end
            
        case 'J'    % computing mean via Jeffery
            A = zeros(dims);
            B = sum(spd_matrices,3);
            for i_th = 1:num_spd
                A = A + spd_matrices(:,:,i_th)^ -1 ;
            end
            M = real( (A ^ -0.5) * (A^0.5 * B * A^0.5)^0.5 * (A ^ -0.5) );
            
        case 'L'    % computing mean via LEM
            logm_matrices = zeros(size(spd_matrices));
            for i_th = 1:size(spd_matrices,3)
                logm_matrices(:,:,i_th) = logm(spd_matrices(:,:,i_th));
            end
            mean_logDes = mean(logm_matrices,3);
            M = expm(mean_logDes);
            
    end
    mean_center = M;
end