#!/bin/sh

#base_path="/media/ansary/DriveData/Work/APSIS/datasets/Recognition/"

base_path="/home/apsisdev/ansary/DATASETS/APSIS/Recognition/"

bw_ref="/home/apsisdev/ansary/DATASETS/RAW/bangla_writing/converted/converted/"
bh_ref="/home/apsisdev/ansary/DATASETS/RAW/BN-HTR/"
bs_ref="/home/apsisdev/ansary/DATASETS/RAW/BanglaC/README.txt"
iit_path="/home/apsisdev/Rezwan/cvit_iiit-indic/"
eng_hw_path="/home/apsisdev/ansary/DATASETS/RAW/eng_page/data/"

save_path=$base_path
src_dir="${base_path}source/"
#-----------------------------------------------------------------------------------------------
ds_path="${save_path}datasets/"
iit_bn_ref="${iit_path}bn/vocab.txt"

bw_ds="${ds_path}bw/"
bh_ds="${ds_path}bh/"
bs_ds="${ds_path}bs/"
bn_pr_ds="${ds_path}bangla_printed/"
bn_hr_ds="${ds_path}bangla_handwritten/"
iit_bn_ds="${ds_path}iit.bn/"

en_hr_ds="${ds_path}en_hw/"

#-----------------------------------bangla-----------------------------------------------
#-----------------------------------natrual---------------------------------------------
# python datasets/bangla_writing.py $bw_ref $ds_path
# python datagen.py $bw_ds 
# python datasets/boise_state.py $bs_ref $ds_path
# python datagen.py $bs_ds 
# python datasets/bn_htr.py $bh_ref $ds_path
# python datagen.py $bh_ds 
# python datasets/iit_indic.py $iit_bn_ref $ds_path
# python datagen.py $iit_bn_ds 
#-----------------------------------natrual---------------------------------------------

#-----------------------------------synthetic------------------------------------------
python datagen_synth.py $src_dir "bangla" "printed" $ds_path --num_samples 10000000

#python datagen_synth.py $src_dir "bangla" "handwritten" $ds_path --num_samples 100000
#-----------------------------------synthetic------------------------------------------
#-----------------------------------bangla-----------------------------------------------

#-----------------------------------english-----------------------------------------------
#python datasets/eng_hw.py $eng_hw_path $ds_path
python datagen_synth.py $src_dir "english" "printed" $ds_path --num_samples 10000000 --decomp 0

echo succeeded