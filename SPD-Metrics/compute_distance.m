%%  computing the distance between SPD matrices
% Written by Kai-Xuan Chen (e-mail: kaixuan_chen_jsh@163.com)
% If you find any bug, please contact me.
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



function out_dis = compute_distance(X,Y,type)
    switch type
        case 'A'    % Affine Invariant Riemannian Metric [1,2]
            tmpEig =  eig(X,Y);  
            tmp_dis = sum(log(tmpEig).^2); 
            
        case 'S'    % Stein divergence [1,3]
            t = log(det(0.5*(X+Y))) - 0.5* (log(det(X)) + log(det(Y)));
            if t == inf || isnan(t)
                eigX = eig(X);
                eigX (eigX <= 0) = eps;
                eigY = eig(Y);
                eigY (eigY <= 0) = eps;
                t = real(sum(log(eig(0.5*(X+Y))))) - 0.5 *(real(sum(log(eigX))) + real(sum(log(eigY))));    
            end
            tmp_dis = t;
            
        case 'J'    % Jeffrey divergence [1,4]
            tmp_dis = 0.5*trace(inv(X)*Y)+0.5*trace(inv(Y)*X) - size(X,1);
            
        case 'L'    % Log-Euclidean Metric [1,5]
            tmp_dis = norm((logm(X)-logm(Y)),'fro').^2;
    end
    if tmp_dis <= 1e-15
        out_dis = 0;
    else
        out_dis = sqrt(tmp_dis);
    end
    
end


% [1] More About Covariance Descriptors for Image Set Coding: Log-Euclidean Framework based Kernel Matrix Representation[C]//Proceedings of the IEEE International Conference on Computer Vision Workshops. 2019: 0-0.
% [2] A riemannian framework for tensor computing,International Journal of computer vision 66 (1) (2006) 41每66.
% [3] A new metric on the manifold of kernel matrices with application to matrix geometric means. Advances in neural information processing systems, 2012,500 pp. 144每152.
% [4] An affine invariant tensor dissimilarity measure and its applications to tensor-valued image segmentation. Proceedings of the 2004 IEEE Computer Society Conference on, Vol. 1, IEEE, 2004, pp. I每I.
% [5] Geometric means in a novel vector space structure on symmetric positive-definite matrices, SIAM journal on matrix analysis and applications 29 (1) (2007) 328每347.
