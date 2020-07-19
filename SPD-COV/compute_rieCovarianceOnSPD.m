%%  compute covariance via four metrics on the Symmetric Positive Definite(SPD) manifold. s)
% Points are two sets of SPD matrices: X(:,:,1), ..., X(:,:,n) & Y(:,:,1), ..., Y(:,:,n)
% Rewritten by Kai-Xuan Chen (e-mail: kaixuan_chen_jsh@163.com),If you find any bugs, please contact me.
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
%     X : X(:,:,1), ..., X(:,:,n) are n SPD matrix
%     Y : Y(:,:,1), ..., Y(:,:,n) are n SPD matrix
%     type_metric: Riemannian metric used for SPD manifold('A':AIRM, 'S':Stein, 'J':Jeffrey, 'L':LEM. )
% output
%     rieCovariance: the Riemannian covariance computed by this function



function rie_covariance = compute_rieCovarianceOnSPD(X,Y,type_metric)

    [~,~,num_X] = size(X);
    [~,~,num_Y] = size(Y);
    if num_X ~= num_Y
        error('THE NUMBER OF TWO SETS ARE NOT INCONSISTENT!');
    else
        num_samples = num_X;
    end
    mean_X = compute_riemannian_mean(X,type_metric);
    mean_Y = compute_riemannian_mean(Y,type_metric);
    
    temp_covariance = 0;
    for i = 1:num_samples
        rie_ldv_xi = compute_ldv(mean_X, X(:,:,i), type_metric);
        rie_ldv_xi = trans_vectorSignNorm(rie_ldv_xi,'s');
        rie_ldv_yi = compute_ldv(mean_Y, Y(:,:,i), type_metric);
        rie_ldv_yi = trans_vectorSignNorm(rie_ldv_yi,'s');
        
        temp_covariance = temp_covariance + rie_ldv_xi'*rie_ldv_yi;
        
    end

    rie_covariance = temp_covariance/(num_samples - 1);
    
end


function out_V = trans_vectorSignNorm(input_V,type_norm_ldv)
    out_V = zeros(size(input_V));
    for i = 1:size(input_V,2)
        temp_input_V = input_V(:,i);
        if strcmp(type_norm_ldv,'s')
            temp_out_V =  sign(temp_input_V) .* abs(temp_input_V).^ (0.5);
        elseif strcmp(type_norm_ldv,'n')
            temp_out_V = temp_input_V ./norm(temp_input_V,2);
        elseif strcmp(type_norm_ldv,'m')
            temp_input_V =  sign(temp_input_V) .* abs(temp_input_V).^ (0.5);
            temp_out_V = temp_input_V ./norm(temp_input_V,2);
        else
            temp_out_V = temp_input_V;
        end
        out_V(:,i) = temp_out_V;
    end
end