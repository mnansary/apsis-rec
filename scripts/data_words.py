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
import json
import pandas as pd 

from tqdm import tqdm
from ast import literal_eval

from coreLib.utils import LOG_INFO
from coreLib.dataset import DataSet
from coreLib.synthetic import createFontFacedWords,createWords
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    data_path   =   args.data_path
    save_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    num_samples =   int(args.num_samples)
    dict_max_len=   int(args.dict_max_len)
    pad_height  =   int(args.pad_height)
    # dataset object
    ds=DataSet(data_dir=data_path,check_english=True)
    # create img_path in df
    ds.english.graphemes.df["img_path"]=ds.english.graphemes.df.filename.progress_apply(lambda x:os.path.join(ds.english.graphemes.dir,f"{x}.bmp"))
    ds.symbols.df["img_path"]=ds.symbols.df.filename.progress_apply(lambda x:os.path.join(ds.symbols.dir,f"{x}.bmp"))
    df=pd.concat([ds.english.graphemes.df,ds.symbols.df])
    LOG_INFO("Creating english words data")
    createWords(iden="en.words",
                df=df,
                save_dir=save_path,
                img_dim=(img_height,img_width),
                comp_dim=img_height,
                pad_height=pad_height,
                top_exts=[],
                bot_exts=[],
                dictionary=ds.english.dictionary,
                num_samples=num_samples)
    
    LOG_INFO("Creating english fontfaced data")
    createFontFacedWords(iden="font.en.words",
                        save_dir=save_path,
                        all_fonts=ds.english.all_fonts,
                        img_dim=(img_height,img_width),
                        comp_dim=img_height,
                        num_samples=num_samples,
                        dict_max_len=dict_max_len,
                        valid_graphemes=ds.english.ffvocab)
    
    LOG_INFO("Creating english raw data")
    
    ds=DataSet(data_dir=data_path,check_english=False)
    LOG_INFO("Creating bangla fontfaced data")
    createFontFacedWords(iden="font.bn.words",
                        save_dir=save_path,
                        all_fonts=ds.bangla.all_fonts,
                        img_dim=(img_height,img_width),
                        comp_dim=img_height,
                        num_samples=num_samples,
                        dict_max_len=dict_max_len,
                        valid_graphemes=ds.bangla.ffvocab)
    

    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Training: english Synthetic (numbers and graphemes) Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the source data folder ")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width dimension of word images: default=512")
    parser.add_argument("--pad_height",required=False,default=20,help ="pad height for each grapheme for alignment correction: default=20")
    parser.add_argument("--num_samples",required=False,default=100000,help ="number of samples to create when not using dictionary:default=100000")
    parser.add_argument("--dict_max_len",required=False,default=30,help ="max number of graphemes in a word:default=20")
    
    args = parser.parse_args()
    main(args)
    
    