%% Riemannian k-means on the Symmetric Positive Definite(SPD) manifold. (https://github.com/Kai-Xuan/MyNote/tree/master/ML/SPD-Means)
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
%     k_centers : the number of the centers
%     type_metric: Riemannian metric used for SPD manifold ('A':AIRM, 'S':Stein, 'J':Jeffrey, 'L':LEM. )
                    
% output
%     index_samples: the label of samples
%     centers_samples: the k centers computed by this function


function [index_samples, centers_samples] = compute_riemannian_kmeans(spd_matrices, k_centers, type_metric, max_iter)
    
    if (nargin < 4)
        max_iter = 15;
    end
    num_sample = size(spd_matrices,3);
    init_centerInd = randperm(num_sample,k_centers);
    init_centers = spd_matrices(:,:,init_centerInd);      
    temp_centers = zeros(size(init_centers));
    
    for iter_th = 1:max_iter
        
        dis_matrix = compute_disMatrix(spd_matrices,init_centers,type_metric);
        [~, temp_index] = min(dis_matrix, [], 2);
        
        for k = 1:k_centers        
            current_centerSamples = find(temp_index == k );   % nonzero elements for a cluster
            if (length(current_centerSamples) > 1)  
                temp_centers(:,:,k) = compute_riemannian_mean(spd_matrices(:,:,current_centerSamples),type_metric);                 
            elseif (length(current_centerSamples) == 1)               
                temp_centers(:,:,k) = spd_matrices(:,:,current_centerSamples);
            else
                disp(['No assignment to a cluster:' int2str(k)]);
            end
        end
        temp2= find(temp_centers - init_centers);
        if (isempty(temp2) && iter_th > 1)    % No change in centers
            break;
        end
        init_centers = temp_centers;
    end
    
    index_samples = temp_index;
    centers_samples = temp_centers;
    
end


function dis_matrix = compute_disMatrix(spd_matrices,center_matrices,type_metric)
    dis_matrix = zeros(size(spd_matrices,3), size(center_matrices,3));
    for i = 1:size(spd_matrices,3)
        for j = 1:size(center_matrices,3)
            dis_matrix(i,j) = compute_distance(spd_matrices(:,:,i), center_matrices(:,:,j), type_metric);            
        end
    end
end


