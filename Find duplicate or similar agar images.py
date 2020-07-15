# -*- coding: utf-8 -*-
"""
Find duplicate/similar agar images


Created on Mon Mar 16 14:25:19 2020

@author: lauri
"""

#################################################
#### Check for duplicate images using hashes ####
#################################################
import glob
import os
import imagehash
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Get paths to all images in NOVODATA
glob_pattern = os.path.join("C:\\Users\\lauri\\Desktop\\DataPreprocessing\\Data\\rl_images\\Rejected - 2nd half",'*')
glob_pattern = os.path.join("C:\\Users\\lauri\\Desktop\\get images\\images", '*', '*.jpg')
glob_pattern = os.path.join("C:\\Users\\lauri\\Desktop\\Images_256", '*', '*', '*.jpg')
glob_pattern = os.path.join("C:\\Users\\lauri\\Desktop\\generated_images\\generated_images", '*.jpg')
image_filenames=sorted(glob.glob(glob_pattern))

"""
image_filenames=list()
glob_pattern = os.path.join("C:\\Users\\lauri\\Desktop\\Images_256", 'Validation', '*', '*.jpg')
image_filenames.extend(sorted(glob.glob(glob_pattern)))
glob_pattern = os.path.join("C:\\Users\\lauri\\Desktop\\Images_256", 'Test', '*', '*.jpg')
image_filenames.extend(sorted(glob.glob(glob_pattern)))
glob_pattern = os.path.join("C:\\Users\\lauri\\Desktop\\Images_256", 'Training', '*', '*.jpg')
image_filenames.extend(sorted(glob.glob(glob_pattern)))
"""

test_files=list()
#test_files.append(r"C:\Users\lauri\Desktop\get images\images\BacSpot test 2_Sep02 max load\reject_24a31505-5898-4dd7-b472-4a35c812020f_80_rl.jpg")
#test_files.append(r"C:\Users\lauri\Desktop\get images\images\BacSpot test 2_Sep02 max load 2\reject_972822ee-3ce3-485d-9b41-c1915e28630b_74_rl.jpg")
#test_files.append(r"C:\Users\lauri\Desktop\get images\images\BacSpot test 2_Sep02 max load 2\reject_972822ee-3ce3-485d-9b41-c1915e28630b_71_rl.jpg")
#test_files.append(r"C:\Users\lauri\Desktop\DataPreprocessing\get images\images\BacSpot test 2_Sep02 crash repair_Sep02_crash 2 reload\reject_a13dede7-f4e4-4716-9830-c9194db8b002_1_rl.jpg")
#test_files.append(r"C:\Users\lauri\Desktop\DataPreprocessing\get images\images\BacSpot test 2_Sep02 crash repair_Sep02_crash 2 reload\reject_a13dede7-f4e4-4716-9830-c9194db8b002_3_rl.jpg")
#test_files.append(r"C:\Users\lauri\Desktop\DataPreprocessing\get images\images\BacSpot test 2_Aug30 5CFU kant 90Grader x3\reject_aa084d88-b404-4993-868f-f198f3bace35_2_rl.jpg")
#test_files.append(r"C:\Users\lauri\Desktop\DataPreprocessing\get images\images\BacSpot test 2_Aug30 5CFU kant 90Grader x3\reject_aa084d88-b404-4993-868f-f198f3bace35_7_rl.jpg")

test_files.append(r"C:\Users\lauri\Desktop\DataPreprocessing\get images\images\BacSpot test 2_Sep02 crash repair_Sep02_crash 2 reload\reject_a13dede7-f4e4-4716-9830-c9194db8b002_36_rl.jpg")
test_files.append(r"C:\Users\lauri\Desktop\DataPreprocessing\get images\images\BacSpot test 2_Sep 02 max load (80)+1\reject_58b8fa28-97e2-4446-aa45-0508a0b1ba4c_55_rl.jpg")
for file in test_files:
    plt.imshow(Image.open(file))
    plt.show()

import cv2
def load_images(paths):
    images = list()
    for idx,filename in enumerate(paths):
        if idx%100 == 0:
            print(idx)
        image = np.asarray(Image.open(filename))
        images.append(image)
    return images
def colorhist(images):
    chists = list()
    for image in images:
        #if image.shape[2] != 3:
        #    print(image.shape)
        #image_file = Image.open(filename)
        #filename_split = filename.split("\\")[-2:]
        #filename_short = filename_split[0] + "\\" + filename_split[1]
        #filename_short = filename.split("\\")[-1]
        #image_array = np.asarray(image_file)
        chist = cv2.calcHist([image],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
        chist = cv2.normalize(chist,chist).flatten()
        chists.append(chist)
    return chists
def compare_colorhists(chists):
    distances_all = np.zeros((len(chists),len(chists),4))
    for i in range(len(chists)-1):
        if i%100==0:
            print(i)
        for j in range(i,len(chists)):
            #if i!=j:
            cor = cv2.compareHist(chists[i],chists[j],cv2.HISTCMP_CORREL)
            distances_all[i,j,0] = cor
            distances_all[j,i,0] = cor
            chi2 = cv2.compareHist(chists[i],chists[j],cv2.HISTCMP_CORREL)
            distances_all[i,j,1] = chi2 
            distances_all[j,i,1] = chi2
            intersect = cv2.compareHist(chists[i],chists[j],cv2.HISTCMP_INTERSECT)
            distances_all[i,j,2] = intersect
            distances_all[j,i,2] = intersect
            bhatta = cv2.compareHist(chists[i],chists[j],cv2.HISTCMP_BHATTACHARYYA)
            distances_all[i,j,3] = bhatta
            distances_all[j,i,3] = bhatta            
    return distances_all

#dist hist
def chist_dist_hist(distances_all,metric_n,title):
    #dist = distances_all[:,:,metric_n]
    dist_list = list()
    for i in range(len(distances_all)-1):
        if i%100==0:
            print(i)
        for j in range(i+1,len(distances_all)):
            dist_list.append(distances_all[i,j,metric_n])
    plt.hist(dist_list,bins=100)
    plt.title(title)
    plt.show()


rej_2_paths = glob.glob(os.path.join(r'C:\Users\lauri\Desktop\DataPreprocessing\Data\rl_images\Rejected - 2nd half','*'))
rej_1_paths = glob.glob(os.path.join(r'C:\Users\lauri\Desktop\DataPreprocessing\Data\rl_images\Rejected - 1st half','*'))
rej_1a_paths = glob.glob(os.path.join(r'C:\Users\lauri\Desktop\DataPreprocessing\Data\rl_images\Rejected - 1st half - maybes','*'))
acc_paths = glob.glob(os.path.join(r'C:\Users\lauri\Desktop\DataPreprocessing\Data\rl_images\Accepted','*'))

def show_dist_hist_for_images(path,title):
    all_images = load_images(path)
    all_chists = colorhist(all_images)
    distances_all = compare_colorhists(all_chists)
    chist_dist_hist(distances_all,0,title)
    chist_dist_hist(distances_all,1,title)
    chist_dist_hist(distances_all,2,title)
    chist_dist_hist(distances_all,3,title)

show_dist_hist_for_images(rej_2_paths,'Positive - part 1')
show_dist_hist_for_images(acc_paths,'Negative')




def hash_images(paths,
                hash_size=8,
                avg_RGB=True,
                a_Hash=True,
                p_Hash=False,
                d_Hash=True,
                w_Hash=False,
                color_Hash=True,
                register_copies=False):    
    count=0
    d=dict() #Key: the images hashes. Value: The image-files that results in this hash

    for filename in paths:
        image_file = Image.open(filename)
        filename_split = filename.split("\\")[-2:]
        filename_short = filename_split[0] + "\\" + filename_split[1]
        filename_short = filename.split("\\")[-1]
        image_array = np.asarray(image_file)
        
        
        
        
        #hex=hashlib.md5(image_array).hexdigest()
        hex=filename_short
        if hex not in d:
            d[hex]={"filename":[filename_short]}
            if avg_RGB:
                d[hex]["avg_RGB"]=np.mean(image_array,axis=(0,1))
            if a_Hash:
                ahash=imagehash.average_hash(image_file,hash_size=hash_size)
                d[hex]["a_Hash"]=str(ahash)
            if p_Hash:
                phash=imagehash.phash(image_file,hash_size=hash_size)
                d[hex]["p_Hash"]=str(phash)
            if d_Hash:
                dhash=imagehash.dhash(image_file,hash_size=hash_size)
                d[hex]["d_Hash"]=str(dhash)
            if w_Hash:
                whash=imagehash.whash(image_file,hash_size=hash_size)
                d[hex]["w_Hash"]=str(whash)
            if color_Hash:
                colorhash=imagehash.colorhash(image_file,hash_size=hash_size)
                d[hex]['color_Hash']=str(colorhash)
        elif register_copies:
            d[hex]['filename'].extend([filename_short])
        if count%100 == 0:
            print(f"{count+1} of {len(paths)}")
        count+=1
    return d
d=hash_images(test_files,
              avg_RGB=False,
              a_Hash=True,
              d_Hash=True,
              w_Hash=True,
              p_Hash=True,
              color_Hash=True,
              register_copies=True)

for key in d.keys():
    print(f"{key}:\t{d[key]['a_Hash']}")
for key in d.keys():
    print(f"{key}:\t{d[key]['d_Hash']}")
for key in d.keys():
    print(f"{key}:\t{d[key]['w_Hash']}")
for key in d.keys():
    print(f"{key}:\t{d[key]['p_Hash']}")


def get_duplicates(d):
    #Get only the duplicates
    dups=dict()#duplicates
    count=0
    for key in d:
        if len(d[key]['filename'])>1:
            dups[key]=d[key]['filename']
            #print(d[key]['filename'])
        if count%100==0:
            print(f"{count+1} of 1958")
        count+=1

    #Convert to a dataframe, for easy analysis
    df=pd.DataFrame(dups)
    df=df.transpose()
    return df
duplicates=get_duplicates(d)

d['0631a3d3a18d4a91f5632979480b9576']['a_Hash']
d['0158f26a33946c4c39a3e0ea00b82ba6']['a_Hash']    
    
"""
duplicates_pairs=dict()
count=0
for i in duplicates.iterrows():
    count+=1
    img1=i[1][0].split("\\")[-1]
    img2=i[1][1].split("\\")[-1]
    
    print(used_images.loc[img1])
    print(used_images.loc[img2])
    same_class  = used_images.loc[img1][0]==used_images.loc[img2][0]
    same_usage  = used_images.loc[img1][1]==used_images.loc[img2][1]
    same_folder = used_images.loc[img1][2]==used_images.loc[img2][2]
    
    duplicates_pairs[count] = [img1,
                               img2,
                               used_images.loc[img1][2],
                               'same folder' if same_folder else used_images.loc[img2][2],
                               used_images.loc[img1][0],
                               'same class' if same_class else used_images.loc[img2][0],
                               used_images.loc[img1][1],
                               'same usage' if same_class else used_images.loc[img2][1]]
duplicates_pairs=pd.DataFrame(duplicates_pairs).transpose()
"""

def measure_hash_sim(d):
    ## Measure the similarity between the hash outputs
    avg_RGB = True if 'avg_RGB' in d[list(d.keys())[0]] else False
    a_Hash  = True if 'a_Hash'  in d[list(d.keys())[0]] else False
    p_Hash  = True if 'p_Hash'  in d[list(d.keys())[0]] else False
    d_Hash  = True if 'd_Hash'  in d[list(d.keys())[0]] else False
    w_Hash  = True if 'w_Hash'  in d[list(d.keys())[0]] else False
    
    k1_count=0
    k2_count=0
    d_copy=d.copy()
    
    #df_all=pd.DataFrame(np.empty((len(d.keys()),len(d.keys()))))
    #df_all.index=sorted(d.keys())
    #df_all.columns=sorted(d.keys())
    
    pairs_all=list()
    sims_all =list()
    
    for key1 in d:
        k1_count+=1
        if k1_count%50==0:
            print(k1_count)
            print(k2_count)
            print()
        del d_copy[key1]
        k2_count=k1_count
        for key2 in d_copy:
            d_sims=dict()
            k2_count+=1
            
            #print(f"\n\nj={k1_count}, k={k2_count}")
            #print(f"{d[key1]['filename']}\n{d_copy[key2]['filename']}\n")
            total=0
            if avg_RGB:
                RGB_sim=np.abs(d[key1]['avg_RGB']-d_copy[key2]['avg_RGB'])
                d_sims['RGB_sim']=RGB_sim
                total+=RGB_sim
                #print(RGB_sim)
            if a_Hash:
                ahash_sim=0
                for i in range(16):
                    if d[key1]["a_Hash"][i]!=d_copy[key2]["a_Hash"][i]:
                        ahash_sim+=1  
                d_sims['a_Hash']=ahash_sim  
                total+=ahash_sim
                #print(ahash_sim)
            if p_Hash:
                phash_sim=0
                for i in range(16):
                    if d[key1]["p_Hash"][i]!=d_copy[key2]["p_Hash"][i]:
                        phash_sim+=1
                d_sims['p_Hash']=phash_sim
                total+=phash_sim
                #print(phash_sim)
            if d_Hash:
                dhash_sim=0
                for i in range(16):
                    if d[key1]["d_Hash"][i]!=d_copy[key2]["d_Hash"][i]:
                        dhash_sim+=1
                d_sims['d_Hash']=dhash_sim
                total+=dhash_sim
                #print(dhash_sim)
            if w_Hash:
                whash_sim=0
                for i in range(16):
                    if d[key1]["w_Hash"][i]!=d_copy[key2]["w_Hash"][i]:
                        whash_sim+=1
                d_sims['w_Hash']=whash_sim
                total+=whash_sim
                #print(whash_sim)
            #df_all.loc[key1,key2]=total
            pairs_all.append([key1,key2])
            sims_all.append(total)
        #k2_count=0    
    return pairs_all, sims_all
            

pairs_all,sims_all=measure_hash_sim(d)

pair_1=[pair[0] for pair in pairs_all]
pair_2=[pair[1] for pair in pairs_all]
df_all=pd.DataFrame()
df_all['p1']=pair_1
df_all['p2']=pair_2
df_all['sim']=sims_all
df_sorted=df_all.sort_values(by='sim',ascending=False)
df_subset=df_sorted.iloc[:100]


np.max(df_all)
columns=df_all.columns
indexs =df_all.index
df_all[columns[1]]
argmaxs=list()
for i in range(len(df_all)):
    
    
    print(df_all.iloc[i,:])
    break




for row in df_all.iterrows():
    print(row[1])
    print(np.max(row[1]))
    argmax=np.argmax(row[1])
    print(row[1][argmax])
    break

