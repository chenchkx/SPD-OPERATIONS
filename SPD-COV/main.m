%%  compute covariance via four metrics on the Symmetric Positive Definite(SPD) manifold. 
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
for i =1:50
    feature_matri_x = rand(10,100);
    feature_matri_y = rand(10,100)+1;
    cov_matrix_x = cov(feature_matri_x');  
    cov_matrix_y = cov(feature_matri_y');  
    spd_matrices_x(:,:,i) = cov_matrix_x + 0.001*trace(cov_matrix_x)*eye(size(cov_matrix_x));
    spd_matrices_y(:,:,i) = cov_matrix_y + 0.001*trace(cov_matrix_y)*eye(size(cov_matrix_y));
end

% compute covariance on the SPD manifold via AIRM
covariance_A = compute_rieCovarianceOnSPD(spd_matrices_x,spd_matrices_y,'A');

% compute covariance on the SPD manifold via Stein
covariance_S = compute_rieCovarianceOnSPD(spd_matrices_x,spd_matrices_y,'S');

% compute covariance on the SPD manifold via Jeffrey
covariance_J = compute_rieCovarianceOnSPD(spd_matrices_x,spd_matrices_y,'J');

% compute covariance on the SPD manifold via LEM
covariance_L = compute_rieCovarianceOnSPD(spd_matrices_x,spd_matrices_y,'L');


