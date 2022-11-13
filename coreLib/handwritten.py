# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import cv2
import numpy as np
import random
import pandas as pd 
from tqdm import tqdm
from .utils import *
from .dataset import DataSet
tqdm.pandas()
#--------------------
# helpers
#--------------------
def createImgFromComps(df,comps,pad):
    '''
        creates a synthetic image from given comps
        args:
            df      :       dataframe holding : "filename","label","img_path"
            comps   :       list of graphemes
            pad     :       pad class:
                                no_pad_dim
                                single_pad_dim
                                double_pad_dim
                                top
                                bot
        returns:
            non-pad-corrected raw binary image
    '''
    # get img_paths
    img_paths=[]
    for idx,comp in enumerate(comps):
        cdf=df.loc[df.label==comp]
        cdf=cdf.sample(frac=1)
        cdf.reset_index(drop=True,inplace=True)
        img_paths.append(cdf.iloc[0,2])
    
    # get images
    imgs=[cv2.imread(img_path,0) for img_path in img_paths]
    # alignment of component
    ## flags
    tp=False
    bp=False
    comp_heights=["" for _ in comps]
    for idx,comp in enumerate(comps):
        if any(te.strip() in comp for te in pad.top):
            comp_heights[idx]+="t"
            tp=True
        if any(be in comp for be in pad.bot):
            comp_heights[idx]+="b"
            bp=True

    # image construction based on height flags
    '''
    class pad:
        no_pad_dim      =(comp_dim,comp_dim)
        single_pad_dim  =(int(comp_dim+pad_height),int(comp_dim+pad_height))
        double_pad_dim  =(int(comp_dim+2*pad_height),int(comp_dim+2*pad_height))
        top             =top_exts
        bot             =bot_exts
        height          =pad_height  
    '''
    cimgs=[]
    for img,hf in zip(imgs,comp_heights):
        if hf=="":
            img=cv2.resize(img,pad.no_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if tp:
                h,w=img.shape
                top=np.ones((pad.height,w))*255
                img=np.concatenate([top,img],axis=0)
            if bp:
                h,w=img.shape
                bot=np.ones((pad.height,w))*255
                img=np.concatenate([img,bot],axis=0)
        elif hf=="t":
            img=cv2.resize(img,pad.single_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if bp:
                h,w=img.shape
                bot=np.ones((pad.height,w))*255
                img=np.concatenate([img,bot],axis=0)

        elif hf=="b":
            img=cv2.resize(img,pad.single_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if tp:
                h,w=img.shape
                top=np.ones((pad.height,w))*255
                img=np.concatenate([top,img],axis=0)
        elif hf=="bt" or hf=="tb":
            img=cv2.resize(img,pad.double_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        
        cimgs.append(img)

    img=np.concatenate(cimgs,axis=1)
    return img


    
def createRandomDictionary(valid_graphemes,num_samples):
    '''
        creates a randomized dictionary
        args:
            valid_graphemes :       list of graphemes that can be used to create a randomized dictionary 
            num_samples     :       number of data to be created if no dictionary is provided
        returns:
            a dictionary dataframe with "word" and "graphemes"
    '''
    word=[]
    graphemes=[]
    for _ in tqdm(range(num_samples)):
        len_word=random.choices(population=[2,3,4,5,6,7],weights=[0.1,0.2,0.2,0.2,0.2,0.1],k=1)[0]
        _graphemes=[]
        for _ in range(len_word):
            # grapheme
            _graphemes.append(random.choice(valid_graphemes))
        graphemes.append(_graphemes)
        word.append("".join(_graphemes))
    df=pd.DataFrame({"word":word,"graphemes":graphemes})
    return df 



#--------------------
# ops
#--------------------
def createSyntheticData(iden,
                        save_dir,
                        data_dir,
                        language,
                        num_samples=100000,
                        comp_dim=64,
                        pad_height=20):
    '''
        creates: 
            * handwriten word image
            * a dataframe/csv that holds word level groundtruth    
    '''
    #---------------
    # processing
    #---------------
    save_dir=create_dir(save_dir,iden)
    LOG_INFO(save_dir)
    # save_paths
    class save:    
        img=create_dir(save_dir,"images")
        csv=os.path.join(save_dir,"data.csv")
        txt=os.path.join(save_dir,"data.txt")
    
    ds=DataSet(data_dir,language.iden)
    valid_graphemes=ds.valid_graphemes
    
    # pad
    class pad:
        no_pad_dim      =(comp_dim,comp_dim)
        single_pad_dim  =(int(comp_dim+pad_height),int(comp_dim+pad_height))
        double_pad_dim  =(int(comp_dim+2*pad_height),int(comp_dim+2*pad_height))
        top             =language.top_exts
        bot             =language.bot_exts
        height          =pad_height   

    # save data
    dictionary=createRandomDictionary(valid_graphemes,num_samples)
    # dataframe vars
    filepaths=[]
    words=[]
    fiden=0
    # loop
    for idx in tqdm(range(len(dictionary))):
        try:
            comps=dictionary.iloc[idx,1]
            img_height=random.randint(24,128)
            # image
            img=createImgFromComps(df=ds.df,comps=comps,pad=pad)
            img=255-img
            h,w=img.shape
            w_new=int(img_height* w/h) 
            img=cv2.resize(img,(w_new,img_height))
            img=255-img
            img=paper_noise(img)
            # save
            fname=f"{fiden}.png"
            cv2.imwrite(os.path.join(save.img,fname),img)
            filepaths.append(os.path.join(save.img,fname))
            word="".join(comps)
            words.append(word)
            fiden+=1
            with open(save.txt,"a+") as f:
                f.write(f"{os.path.join(save.img,fname)}\t{word}\n")
            
        except Exception as e:
           LOG_INFO(e)
    df=pd.DataFrame({"filepath":filepaths,"word":words})
    df.to_csv(os.path.join(save.csv),index=False)