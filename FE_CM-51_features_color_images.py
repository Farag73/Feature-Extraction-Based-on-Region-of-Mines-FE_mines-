# paper Title: (A new feature extraction approach of medical image based on data distribution skew)
# Journal    : Neuroscience Informatics 
# puplisher  : Elsevier 
# Auther name: Farag Kuwil
# Available online 5 August 2022
# URL        :  https://www.sciencedirect.com/science/article/pii/S2772528622000590

import pywt
import pandas as pd
import cv2
import numpy as np
import imageio as iio
import os
import random
from pathlib import Path
import glob
import timeit
from PIL import Image
from scipy import stats 


start = timeit.default_timer()
cv_img = [];j=0;    feature = pd.DataFrame()

valid_images = [".jpg",".gif",".png",".tga"]
for im in glob.glob('D:\papers\RGB_ALL\RGB\dataset\eyes\class_1/*.png'):  #  or *.jpg
    img= cv2.imread(im)
   
    img= cv2.imread(im)
    med_r =  np.median(img);
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    med_g =  np.median(gray);
    ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    med_bw =  np.median(bw);

    (cD2, cA1) = pywt.dwt(gray, 'db1')
    (cD1, cA2) = pywt.dwt(img, 'db1')
    (cD3, cA3) = pywt.dwt(bw, 'db1')
    
    med_cA1 =  np.median(cA1);
    med_cA2 =  np.median(cA2);
    med_cA3 =  np.median(cA3);

    Q1_g = gray[gray<med_g]; Q3_g = gray[gray>=med_g];   
    v1 = np.mean(Q1_g);  v2 = np.std(Q1_g);  v3=v2/v1 ; 
    v4 = np.mean(Q3_g);  v5 = np.std(Q3_g);  v6=v5/v4 ;   
    v7 = np.mean(gray);  v8 = np.std(gray);  v9=v8/v7 ;   

    Q1_r = img[img<med_r]; Q3_r = img[img>=med_r];   
    v10 = np.mean(Q1_r);  v11 = np.std(Q1_r);  v12=v2/v1 ; 
    v13 = np.mean(Q3_r);  v14 = np.std(Q3_r);  v15=v14/v13 ;   
    v16 = np.mean(img);  v17 = np.std(img);  v18=v17/v16 ;   

    Q1_bw = bw[bw<med_cA1]; Q3_bw = bw[bw>=med_cA1];   
    v19 = np.mean(Q3_bw);  v20 = np.std(Q3_bw);  v21=v20/v19 ;   
    v22 = np.mean(bw);  v23 = np.std(bw);  v24=v23/v22 ;     

    Q1_cA1 = cA1[cA1<med_cA1]; Q3_cA1 = cA1[cA1>=med_cA1];   
    v25 = np.mean(Q1_cA1);  v26 = np.std(Q1_cA1);  v27=v26/v25 ; 
    v28 = np.mean(Q3_cA1);  v29 = np.std(Q3_cA1);  v30=v29/v28 ;   
    v31 = np.mean(cA1);  v32 = np.std(cA1);  v33=v32/v31 ;     

    Q1_cA2 = cA2[cA2<med_cA2]; Q3_cA2 = cA2[cA2>=med_cA2];   
    v34 = np.mean(Q1_cA2);  v35 = np.std(Q1_cA2);  v36=v35/v34 ; 
    v37 = np.mean(Q3_cA2);  v38 = np.std(Q3_cA2);  v39=v38/v37 ;   
    v40 = np.mean(cA2);     v41 = np.std(cA2);     v42=v41/v40 ;     

    Q1_cA3 = cA3[cA3<med_bw]; Q3_cA3 = cA3[cA3>=med_cA3];   
    v43 = np.mean(Q1_cA3);  v44 = np.std(Q1_cA3);  v45=v44/v43 ; 
    v46 = np.mean(Q3_cA3);  v47 = np.std(Q3_cA3);  v48=v47/v46 ;   
    v49 = np.mean(cA3);     v50 = np.std(cA3);     v51=v50/v49 ;   
    
    feature.loc[0,j]  = v1;    feature.loc[1,j]  = v2;    feature.loc[2,j]  = v3;
    feature.loc[3,j]  = v4;    feature.loc[4,j]  = v5;    feature.loc[5,j]  = v6;
    feature.loc[6,j]  = v7;    feature.loc[7,j]  = v8;    feature.loc[8,j]  = v9;
    
    feature.loc[9,j]  = v10;    feature.loc[10,j]  = v11;    feature.loc[11,j]  = v12;
    feature.loc[12,j]  = v13;    feature.loc[13,j]  = v14;    feature.loc[14,j]  = v15;
    feature.loc[15,j]  = v16;    feature.loc[16,j]  = v17;    feature.loc[17,j]  = v18;

    feature.loc[18,j]  = v19;    feature.loc[19,j]  = v20;    feature.loc[20,j]  = v21;
    feature.loc[21,j]  = v22;    feature.loc[22,j]  = v23;    feature.loc[23,j]  = v24;
    
    feature.loc[24,j]  = v25;    feature.loc[25,j]  = v26;    feature.loc[26,j]  = v27;
    feature.loc[27,j]  = v28;    feature.loc[28,j]  = v29;    feature.loc[29,j]  = v30;
    feature.loc[30,j]  = v31;    feature.loc[31,j]  = v32;    feature.loc[32,j]  = v33;

    feature.loc[33,j]  = v34;    feature.loc[34,j]  = v35;    feature.loc[35,j]  = v36;
    feature.loc[36,j]  = v37;   feature.loc[37,j]  = v38;    feature.loc[38,j]  = v39;
    feature.loc[39,j]  = v40;   feature.loc[40,j]  = v41;    feature.loc[41,j]  = v42;

    feature.loc[42,j]  = v43;   feature.loc[43,j]  = v44;    feature.loc[44,j]  = v45;
    feature.loc[45,j]  = v46;   feature.loc[46,j]  = v47;    feature.loc[47,j]  = v48;
    feature.loc[48,j]  = v49;   feature.loc[49,j]  = v50;    feature.loc[50,j]  = v51;

    feature.loc[51,j] = 1;    
    j=j+1; 
#Extract features from class 2
    
for im in glob.glob('D:\\papers\\RGB_ALL\\RGB\\dataset\\eyes\\class_0/*.png'):  #  or *.jpg
    img= cv2.imread(im)
   
    img= cv2.imread(im)
    med_r =  np.median(img);
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    med_g =  np.median(gray);
    ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    med_bw =  np.median(bw);
    (cD2, cA1) = pywt.dwt(gray, 'db1')
    (cD1, cA2) = pywt.dwt(img, 'db1')
    (cD3, cA3) = pywt.dwt(bw, 'db1')
    
    med_cA1 =  np.median(cA1);
    med_cA2 =  np.median(cA2);
    med_cA3 =  np.median(cA3);

    Q1_g = gray[gray<med_g]; Q3_g = gray[gray>=med_g];   
    v1 = np.mean(Q1_g);  v2 = np.std(Q1_g);  v3=v2/v1 ; 
    v4 = np.mean(Q3_g);  v5 = np.std(Q3_g);  v6=v5/v4 ;   
    v7 = np.mean(gray);  v8 = np.std(gray);  v9=v8/v7 ;   

    Q1_r = img[img<med_r]; Q3_r = img[img>=med_r];   
    v10 = np.mean(Q1_r);  v11 = np.std(Q1_r);  v12=v2/v1 ; 
    v13 = np.mean(Q3_r);  v14 = np.std(Q3_r);  v15=v14/v13 ;   
    v16 = np.mean(img);  v17 = np.std(img);  v18=v17/v16 ;   

    Q1_bw = bw[bw<med_cA1]; Q3_bw = bw[bw>=med_cA1];   
    v19 = np.mean(Q3_bw);  v20 = np.std(Q3_bw);  v21=v20/v19 ;   
    v22 = np.mean(bw);  v23 = np.std(bw);  v24=v23/v22 ;     

    Q1_cA1 = cA1[cA1<med_cA1]; Q3_cA1 = cA1[cA1>=med_cA1];   
    v25 = np.mean(Q1_cA1);  v26 = np.std(Q1_cA1);  v27=v26/v25 ; 
    v28 = np.mean(Q3_cA1);  v29 = np.std(Q3_cA1);  v30=v29/v28 ;   
    v31 = np.mean(cA1);  v32 = np.std(cA1);  v33=v32/v31 ;     

    Q1_cA2 = cA2[cA2<med_cA2]; Q3_cA2 = cA2[cA2>=med_cA2];   
    v34 = np.mean(Q1_cA2);  v35 = np.std(Q1_cA2);  v36=v35/v34 ; 
    v37 = np.mean(Q3_cA2);  v38 = np.std(Q3_cA2);  v39=v38/v37 ;   
    v40 = np.mean(cA2);     v41 = np.std(cA2);     v42=v41/v40 ;     

    
    Q1_cA3 = cA3[cA3<med_bw]; Q3_cA3 = cA3[cA3>=med_cA3];   
    v43 = np.mean(Q1_cA3);  v44 = np.std(Q1_cA3);  v45=v44/v43 ; 
    v46 = np.mean(Q3_cA3);  v47 = np.std(Q3_cA3);  v48=v47/v46 ;   
    v49 = np.mean(cA3);     v50 = np.std(cA3);     v51=v50/v49 ;   
    
    feature.loc[0,j]  = v1;    feature.loc[1,j]  = v2;    feature.loc[2,j]  = v3;
    feature.loc[3,j]  = v4;    feature.loc[4,j]  = v5;    feature.loc[5,j]  = v6;
    feature.loc[6,j]  = v7;    feature.loc[7,j]  = v8;    feature.loc[8,j]  = v9;
    
    feature.loc[9,j]  = v10;    feature.loc[10,j]  = v11;    feature.loc[11,j]  = v12;
    feature.loc[12,j]  = v13;    feature.loc[13,j]  = v14;    feature.loc[14,j]  = v15;
    feature.loc[15,j]  = v16;    feature.loc[16,j]  = v17;    feature.loc[17,j]  = v18;

    feature.loc[18,j]  = v19;    feature.loc[19,j]  = v20;    feature.loc[20,j]  = v21;
    feature.loc[21,j]  = v22;    feature.loc[22,j]  = v23;    feature.loc[23,j]  = v24;
    
    feature.loc[24,j]  = v25;    feature.loc[25,j]  = v26;    feature.loc[26,j]  = v27;
    feature.loc[27,j]  = v28;    feature.loc[28,j]  = v29;    feature.loc[29,j]  = v30;
    feature.loc[30,j]  = v31;    feature.loc[31,j]  = v32;    feature.loc[32,j]  = v33;

    feature.loc[33,j]  = v34;    feature.loc[34,j]  = v35;    feature.loc[35,j]  = v36;
    feature.loc[36,j]  = v37;   feature.loc[37,j]  = v38;    feature.loc[38,j]  = v39;
    feature.loc[39,j]  = v40;   feature.loc[40,j]  = v41;    feature.loc[41,j]  = v42;

    feature.loc[42,j]  = v43;   feature.loc[43,j]  = v44;    feature.loc[44,j]  = v45;
    feature.loc[45,j]  = v46;   feature.loc[46,j]  = v47;    feature.loc[47,j]  = v48;
    feature.loc[48,j]  = v49;   feature.loc[49,j]  = v50;    feature.loc[50,j]  = v51;

    feature.loc[51,j] = 2;    
    j=j+1; 
stop = timeit.default_timer();
print('Time: ', stop - start) ; 

df = pd.DataFrame(feature).T
df.to_excel(excel_writer = "D:\papers\RGB_ALL\RGB\Gethub/res_51.xlsx");

 