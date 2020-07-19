%% The code of four metrics for computing distance on the Symmetric Positive Definite(SPD) manifold.  
% Four metrics: 1.Affine Invariant Riemannian Metric(AIRM),
%               2.Stein divergence,
%               3.Jeffrey divergence,
%               4.Log-Euclidean Metric(LEM).
% 
% Written by Kai-Xuan Chen (e-mail: kaixuan_chen_jsh@163.com). If you find any bugs, please contact me. 
% 
% If you find this code useful for your research, we appreciate it very much if you can cite our related works:
% 
% 1. https://github.com/Kai-Xuan/RiemannianCovDs/ 
% Kai-Xuan Chen, Jie-Yi-Ren, Xiao-Jun Wu, Josef Kittler. 
% Covariance Descriptors on a Gaussian Manifold and their Application to Image Set Classification[J]. 
% Pattern Recognition, 2020: 107463.
% 
% 2. https://github.com/Kai-Xuan/ComponentSPD/  
% Kai-Xuan Chen, Xiao-Jun Wu. 
% Component SPD matrices: A low-dimensional discriminative data descriptor for image set classification[J]. 
% Computational Visual Media, 2018, 4(3): 245-252.

clear;  
clc;
feature_matrix1 = rand(3,100);
feature_matrix2 = rand(3,100);
spd_matrix1 = cov(feature_matrix1');    
spd_matrix2 = cov(feature_matrix2');
% spd_matrix1 = spd_matrix1 + 0.001*trace(spd_matrix1)*eye(size(spd_matrix1));
% spd_matrix2 = spd_matrix2 + 0.001*trace(spd_matrix2)*eye(size(spd_matrix2));

%% distance while using AIRM
dis_A = compute_distance(spd_matrix1,spd_matrix2,'A');

%% distance while using Stein
dis_S = compute_distance(spd_matrix1,spd_matrix2,'S');

%% distance while using Jeffrey
dis_J = compute_distance(spd_matrix1,spd_matrix2,'J');

%% distance while using LogED
dis_L = compute_distance(spd_matrix1,spd_matrix2,'L');







%% If you find this code useful for your research, we appreciate it very much if you can cite our related works:
%{
    @article{chen2020covariance,
      title={Covariance Descriptors on a Gaussian Manifold and their Application to Image Set Classification},
      author={Chen, Kai-Xuan and Ren, Jie-Yi and Wu, Xiao-Jun and Kittler, Josef},
      journal={Pattern Recognition},
      pages={107463},
      year={2020},
      publisher={Elsevier}
    }

    @inproceedings{chen2019more,
      title={More About Covariance Descriptors for Image Set Coding: Log-Euclidean Framework based Kernel Matrix Representation},
      author={Chen, Kai-Xuan and Wu, Xiao-Jun and Ren, Jie-Yi and Wang, Rui and Kittler, Josef},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
      pages={0--0},
      year={2019}
    }
%}

                            
                            
