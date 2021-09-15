import numpy as np
import math


def gaussian_func2(x, u, o=50):
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - u, 2)) / (2 * np.power(o, 2))
    return temp1 * np.exp(temp2)


def gauss_smooth2(arr,o):
    hist_num = len(arr)
    vect= np.zeros((hist_num,1))
    for i in range(hist_num):
        vect[i,0]=i
    # gaussian_vect= gaussian_func2(vect,0,1)
    # o=50
    approximate_delta = 3*o     #  when x-u>approximate_delta, e.g., 6*o, the gaussian value is approximately equal to 0.
    gaussian_vect= gaussian_func2(vect,0,o)
    matrix = np.zeros((hist_num,hist_num))
    for i in range(hist_num):
        k=0
        for j in range(i,hist_num):
            if k>approximate_delta:
                continue
            matrix[i][j]=gaussian_vect[j-i] 
            k=k+1  
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i]=matrix[i][i]/2
    # for i in range(hist_num):
    #     for j in range(i):
    #         matrix[i][j]=gaussian_vect[j]     
    xxx = np.dot(matrix,arr)
    return xxx


def st_evaluation(query_features, query_pids, query_camids, query_frames, 
                    gallery_features, gallery_pids, gallery_camids, gallery_frames,
                    score, alpha=5, smooth=50, distribution=None):
    #############################################################
    # print('hi', distribution.shape) # [8,8,5000]
    for i in range(0,8):
        for j in range(0,8):
            # print("gauss "+str(i)+"->"+str(j))
            # gauss_smooth(distribution[i][j])
            distribution[i][j][:]=gauss_smooth2(distribution[i][j][:],smooth)
    eps = 0.0000001
    sum_ = np.sum(distribution,axis=2)
    for i in range(8):
        for j in range(8):
            distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps)
    #############################################################

    interval = 100
    score_st = np.zeros([len(query_features), len(gallery_features)])
    for k in range(len(query_features)):
        print('%d/%d' % (k, len(query_features)))
        for i in range(len(gallery_features)):
            if query_frames[k]>gallery_frames[i]:
                diff = query_frames[k] - gallery_frames[i]
                hist_ = int(diff/interval)
                pr = distribution[query_camids[k]][gallery_camids[i]][hist_]
            else:
                diff = gallery_frames[i] - query_frames[k]
                hist_ = int(diff/interval)
                pr = distribution[gallery_camids[i]][query_camids[k]][hist_]
            score_st[k][i] = pr
    score  = 1/(1+np.exp(-alpha*score))*1/(1+2*np.exp(-alpha*score_st))
    return score
