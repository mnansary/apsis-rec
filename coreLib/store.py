#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import json
import math
import pandas as pd 
import tensorflow as tf
import numpy as np 
import cv2
from ast import literal_eval
from tqdm.auto import tqdm
from .utils import *
tqdm.pandas()
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------

# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_mask(down_factor,mask):
    h,w=mask.shape
    h=h//down_factor
    w=w//down_factor
    mask=cv2.resize(mask,(w,h),fx=0,fy=0,interpolation=cv2.INTER_NEAREST)
    mask=mask.flatten().tolist()
    mask=[int(i) for i in mask]
    return mask

def toTfrecord(df,rnum,rec_path,img_dim,down_factor):
    '''
        args:
            df      :   the dataframe that contains the information to store
            rnum    :   record number
            rec_path:   save_path
            mask_dim:   the dimension of the mask
    '''
    tfrecord_name=f'{rnum}.tfrecord'
    tfrecord_path=os.path.join(rec_path,tfrecord_name) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        
        for idx in range(len(df)):
            # base
            imask   =df.iloc[idx,2]
            img_path=df.iloc[idx,3]
            
            mask=np.zeros(img_dim)
            mask[:,imask:]=1
            mask=get_mask(down_factor,mask)
            try:
                
                # img
                with(open(img_path,'rb')) as fid:
                    image_png_bytes=fid.read()
                # feature desc
                data ={ 'image':_bytes_feature(image_png_bytes)}
                # mask
                data["mask"] =_int64_list_feature(mask)    
                # label
                data["label"]=_int64_list_feature(df.iloc[idx,4]) 

                
                features=tf.train.Features(feature=data)
                example= tf.train.Example(features=features)
                serialized=example.SerializeToString()
                writer.write(serialized)  
            except Exception as e:
                print("# Missing:",img_path)

def createRecords(data,save_path,img_dim,down_factor,tf_size=10240):
    '''
        creates tf records:
        args:
            data        :   either the csv path or a dataframe cols=["filepath","word","datapath","label"]
            save_path   :   location to save tfrecords
    '''
    if type(data)==str:
        data=pd.read_csv(data)
        data.dropna(inplace=True)
        data["label"]=data["label"].progress_apply(lambda x: literal_eval(x))
    data.reset_index(drop=True,inplace=True)
    
    LOG_INFO(f"Creating TFRECORDS No folds:{save_path}")
    for idx in tqdm(range(0,len(data),tf_size)):
        df        =   data.iloc[idx:idx+tf_size] 
        df.reset_index(drop=True,inplace=True) 
        rnum      =   idx//tf_size
        toTfrecord(df,rnum,save_path,img_dim,down_factor)

    
    