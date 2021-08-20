#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
import os 
import pandas as pd 

from tqdm import tqdm

from coreLib.utils import LOG_INFO
from coreLib.dataset import DataSet
from coreLib.synthetic import createWords
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    data_path   =   args.data_path
    save_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    pad_height  =   int(args.pad_height)
    num_samples =   int(args.num_samples)
    # dataset object
    ds=DataSet(data_dir=data_path)
    LOG_INFO("Creating bangla numbers data")
    # create img_path in df
    ds.bangla.numbers.df["img_path"]=ds.bangla.numbers.df.filename.progress_apply(lambda x:os.path.join(ds.bangla.numbers.dir,f"{x}.bmp"))
    ds.symbols.df["img_path"]=ds.symbols.df.filename.progress_apply(lambda x:os.path.join(ds.symbols.dir,f"{x}.bmp"))
    df=pd.concat([ds.bangla.numbers.df,ds.symbols.df])
    LOG_INFO("Creating synthetic bangla numbers data")
    createWords(iden="n.bn",
                df=df,
                save_dir=save_path,
                img_dim=(img_height,img_width),
                comp_dim=img_height,
                pad_height=pad_height,
                top_exts=[],
                bot_exts=[],
                dictionary=None,
                valid_graphemes=ds.bangla.number_values,
                num_samples=num_samples,
                numbers_only=True)            
    
    # dataset object
    ds=DataSet(data_dir=data_path,check_english=True)
    LOG_INFO("Creating english numbers data")
    # create img_path in df
    ds.english.numbers.df["img_path"]=ds.english.numbers.df.filename.progress_apply(lambda x:os.path.join(ds.english.numbers.dir,f"{x}.bmp"))
    ds.symbols.df["img_path"]=ds.symbols.df.filename.progress_apply(lambda x:os.path.join(ds.symbols.dir,f"{x}.bmp"))
    df=pd.concat([ds.english.numbers.df,ds.symbols.df])
    LOG_INFO("Creating synthetic english numbers data")
    createWords(iden="n.en",
                df=df,
                save_dir=save_path,
                img_dim=(img_height,img_width),
                comp_dim=img_height,
                pad_height=pad_height,
                top_exts=[],
                bot_exts=[],
                dictionary=None,
                valid_graphemes=ds.english.number_values,
                num_samples=num_samples,
                numbers_only=True)            
    
    
    
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Training: Synthetic numbers  Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the source data folder ")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--pad_height",required=False,default=20,help ="pad height for each grapheme for alignment correction: default=20")
    parser.add_argument("--img_width",required=False,default=512,help ="width dimension of word images: default=512")
    parser.add_argument("--num_samples",required=False,default=100000,help ="number of samples to create when not using dictionary:default=100000")
    args = parser.parse_args()
    main(args)
    
    