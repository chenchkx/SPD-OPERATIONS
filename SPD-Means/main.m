%%  computing means via four metrics on the Symmetric Positive Definite(SPD) manifold. (https://github.com/Kai-Xuan/MyNote/tree/master/ML/SPD-Means)
% Four metrics: 1.Affine Invariant Riemannian Metric(AIRM),
%               2.Stein divergence,
%               3.Jeffrey divergence,
%               4.Log-Euclidean Metric(LEM).
% 
% Rewritten by Kai-Xuan Chen (e-mail: kaixuan_chen_jsh@163.com),If you find any bugs, please contact me.
% Also, you can find more applications at: https://github.com/Kai-Xuan/RiemannianCovDs/  
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

 

clear;  
clc;

% generate SPD matrices
for i =1:100
    feature_matrix = rand(15,100);
    cov_matrix = cov(feature_matrix');  
    spd_matrices(:,:,i) = cov_matrix + 0.001*trace(cov_matrix)*eye(size(cov_matrix));
end


k = 2; % the number of the centers
%% compute k centers by Riemannian k-means
% computing k-means via AIRM
[index_samples_A, centers_samples_A] = compute_riemannian_kmeans(spd_matrices, k, 'A');

% computing k-means via Stein
[index_samples_S, centers_samples_S] = compute_riemannian_kmeans(spd_matrices, k, 'S');

% computing k-means via Jeffrey
[index_samples_J, centers_samples_J] = compute_riemannian_kmeans(spd_matrices, k, 'J');

% computing k-means via LEM
[index_samples_L, centers_samples_L] = compute_riemannian_kmeans(spd_matrices, k, 'L');



%% If you just compute one center, you can consider the following simple operation
% computing mean via AIRM
mean_center_A = compute_riemannian_mean(spd_matrices, 'A');

% computing mean via Stein
mean_center_S = compute_riemannian_mean(spd_matrices, 'S');

% computing mean via Jeffrey
mean_center_J = compute_riemannian_mean(spd_matrices, 'J');

% computing mean via LEM
mean_center_L = compute_riemannian_mean(spd_matrices, 'L');





                            
                            
