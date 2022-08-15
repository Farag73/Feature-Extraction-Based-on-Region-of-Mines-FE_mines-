# paper Title: (A new feature extraction approach of medical image based on data distribution skew)
# Journal    : Neuroscience Informatics 
# puplisher  : Elsevier 
# Auther name: Farag Kuwil
# Available online 5 August 2022
# URL        :  https://www.sciencedirect.com/science/article/pii/S2772528622000590

import pandas as pd
import cv2
import numpy as np
import imageio as iio
import os
import random
from pathlib import Path
import glob
cv_img = [];i=0;    feature = pd.DataFrame()
import timeit
start = timeit.default_timer()
j=0;
#Extract features from class 1

for im in glob.glob("D:\papers\RGB_ALL\RGB\\dataset\\eyes\\class_0/*.png"):# or *.jpg
    img= cv2.imread(im)
    b, g, r = cv2.split(img)
    med_r =  np.median(r);     med_g =  np.median(g);     med_b =  np.median(b);
    med_i =  np.median(img);
# divide data using median into two regions (Q1_r and Q3_r)   
    Q1_r = r[r<med_r]; Q3_r = r[r>=med_r];   
    v1 = np.mean(Q1_r);  v2 = np.std(Q1_r);  v3=v2/v1 ; 
    v4 = np.mean(Q3_r);  v5 = np.std(Q3_r);  v6=v5/v4 ;   
    v7 = np.mean(r);  v8 = np.std(r);        v9=v8/v7 ;   

    Q1_g = g[g<med_g]; Q3_g = g[g>=med_g];   
    v10 = np.mean(Q1_g);  v11 = np.std(Q1_g);  v12=v11/v10 ; 
    v13 = np.mean(Q3_g);  v14 = np.std(Q3_g);  v15=v14/v13 ;   
    v16 = np.mean(g);  v17 = np.std(g);        v18=v17/v16 ;   
 
    Q1_b = b[b<med_b]; Q3_b = b[b>=med_b];   
    v19 = np.mean(Q1_b);  v20 = np.std(Q1_b);  v21=v20/v19 ; 
    v22 = np.mean(Q3_b);  v23 = np.std(Q3_b);  v24=v23/v22 ;   
    v25 = np.mean(b);  v26 = np.std(b);        v27=v26/v25 ;   

    Q1_i = img[img<med_i];   Q3_i = img[img>=med_i];   
    v28 = np.mean(Q1_i); v29 = np.std(Q1_i);   v30=v29/v28  ;      
    v31 = np.mean(Q3_i); v32 = np.std(Q3_i);   v33=v32/v31 ;        
    v34 = np.mean(img);  v35 = np.std(img);    v36=v35/v34 ;    

    feature.loc[0,j]  = v1;      feature.loc[1,j]  = v2;      feature.loc[2,j]  = v3;
    feature.loc[3,j]  = v4;      feature.loc[4,j]  = v5;      feature.loc[5,j]  = v6;
    feature.loc[6,j]  = v7;      feature.loc[7,j]  = v8;      feature.loc[8,j]  = v9;
    feature.loc[9,j]  = v10;     feature.loc[10,j]  = v11;    feature.loc[11,j]  = v12;
    feature.loc[12,j]  = v13;    feature.loc[13,j]  = v14;    feature.loc[14,j]  = v15;
    feature.loc[15,j]  = v16;    feature.loc[16,j]  = v17;    feature.loc[17,j]  = v18;

    feature.loc[18,j]  = v19;    feature.loc[19,j]  = v20;    feature.loc[20,j]  = v21;
    feature.loc[21,j]  = v22;    feature.loc[22,j]  = v23;    feature.loc[23,j]  = v24;
    feature.loc[24,j]  = v25;    feature.loc[25,j]  = v26;    feature.loc[26,j]  = v27;

    feature.loc[27,j]  = v28;    feature.loc[28,j]  = v29;    feature.loc[29,j]  = v30;
    feature.loc[30,j]  = v31;    feature.loc[31,j]  = v32;    feature.loc[32,j]  = v33;
    feature.loc[33,j]  = v34;    feature.loc[34,j]  = v35;    feature.loc[35,j]  = v36;
    feature.loc[36,j] = 0;    
    j=j+1; 
#Extract features from class 2

for im in glob.glob("D:\\papers\\RGB_ALL\\RGB\\dataset\\eyes\\Class_1/*.png"): # or *.jpg
    img= cv2.imread(im)


    img= cv2.imread(im)
    b, g, r = cv2.split(img)
    med_r =  np.median(r);
    med_g =  np.median(g);
    med_b =  np.median(b);
    med_i =  np.median(img);
    # divide data using median into two regions (Q1_r and Q3_r)   
    Q1_r = r[r<med_r]; Q3_r = r[r>=med_r];   
    v1 = np.mean(Q1_r);  v2 = np.std(Q1_r);  v3=v2/v1 ; 
    v4 = np.mean(Q3_r);  v5 = np.std(Q3_r);  v6=v5/v4 ;   
    v7 = np.mean(r);  v8 = np.std(r);        v9=v8/v7 ;   

    Q1_g = g[g<med_g]; Q3_g = g[g>=med_g];   
    v10 = np.mean(Q1_g);  v11 = np.std(Q1_g);  v12=v11/v10 ; 
    v13 = np.mean(Q3_g);  v14 = np.std(Q3_g);  v15=v14/v13 ;   
    v16 = np.mean(g);  v17 = np.std(g);        v18=v17/v16 ;   
 
    Q1_b = b[b<med_b]; Q3_b = b[b>=med_b];   
    v19 = np.mean(Q1_b);  v20 = np.std(Q1_b);  v21=v20/v19 ; 
    v22 = np.mean(Q3_b);  v23 = np.std(Q3_b);  v24=v23/v22 ;   
    v25 = np.mean(b);  v26 = np.std(b);        v27=v26/v25 ;   

    Q1_i = img[img<med_i];   Q3_i = img[img>=med_i];   
    v28 = np.mean(Q1_i); v29 = np.std(Q1_i);   v30=v29/v28  ;      
    v31 = np.mean(Q3_i); v32 = np.std(Q3_i);   v33=v32/v31 ;        
    v34 = np.mean(img);  v35 = np.std(img);    v36=v35/v34 ;    

    feature.loc[0,j]  = v1;      feature.loc[1,j]  = v2;      feature.loc[2,j]  = v3;
    feature.loc[3,j]  = v4;      feature.loc[4,j]  = v5;      feature.loc[5,j]  = v6;
    feature.loc[6,j]  = v7;      feature.loc[7,j]  = v8;      feature.loc[8,j]  = v9;
    feature.loc[9,j]  = v10;     feature.loc[10,j]  = v11;    feature.loc[11,j]  = v12;
    feature.loc[12,j]  = v13;    feature.loc[13,j]  = v14;    feature.loc[14,j]  = v15;
    feature.loc[15,j]  = v16;    feature.loc[16,j]  = v17;    feature.loc[17,j]  = v18;

    feature.loc[18,j]  = v19;    feature.loc[19,j]  = v20;    feature.loc[20,j]  = v21;
    feature.loc[21,j]  = v22;    feature.loc[22,j]  = v23;    feature.loc[23,j]  = v24;
    feature.loc[24,j]  = v25;    feature.loc[25,j]  = v26;    feature.loc[26,j]  = v27;

    feature.loc[27,j]  = v28;    feature.loc[28,j]  = v29;    feature.loc[29,j]  = v30;
    feature.loc[30,j]  = v31;    feature.loc[31,j]  = v32;    feature.loc[32,j]  = v33;
    feature.loc[33,j]  = v34;    feature.loc[34,j]  = v35;    feature.loc[35,j]  = v36;
    feature.loc[36,j] =1;    
    j=j+1; 

stop = timeit.default_timer();
print('Time: ', stop - start) ; 
df = pd.DataFrame(feature).T;
df.to_excel(excel_writer = "D:\papers\RGB_ALL\RGB\Gethub/res_36.xlsx");
