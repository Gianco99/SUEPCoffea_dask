#!/bin/bash

python datacardMaker_ManualMCStats_OptimalBounds_EMu.py ZH/samples_withSF_nocuts_UL18_dR_HighStats_EMu.py ZH/systs_fullcorr_EMu.py /eos/user/g/gdecastr/SUEPCoffea_dask/Cards_OptimalBounds/EMu_Manual/2018 --rootfile /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/EMu_18_OptimalCut/leadclustertracks --var leadclustertracks --ABCD --year 2018 --region EMu --floatB --doAll --ManualMCStats ;
wait

python datacardMaker_ManualMCStats_OptimalBounds_EMu.py ZH/samples_withSF_nocuts_UL17_dR_HighStats_EMu.py ZH/systs_fullcorr_EMu.py /eos/user/g/gdecastr/SUEPCoffea_dask/Cards_OptimalBounds/EMu_Manual/2017 --rootfile /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/EMu_17_OptimalCut/leadclustertracks --var leadclustertracks --ABCD --year 2017 --region EMu --floatB --doAll --ManualMCStats ;
wait

python datacardMaker_ManualMCStats_OptimalBounds_EMu.py ZH/samples_withSF_nocuts_UL16_dR_HighStats_EMu.py ZH/systs_fullcorr_EMu.py /eos/user/g/gdecastr/SUEPCoffea_dask/Cards_OptimalBounds/EMu_Manual/2016 --rootfile /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/EMu_16_OptimalCut/leadclustertracks --var leadclustertracks --ABCD --year 2016 --region EMu --floatB --doAll --ManualMCStats ;
wait

python datacardMaker_ManualMCStats_OptimalBounds_EMu.py ZH/samples_withSF_nocuts_UL16APV_dR_HighStats_EMu.py ZH/systs_fullcorr_EMu.py /eos/user/g/gdecastr/SUEPCoffea_dask/Cards_OptimalBounds/EMu_Manual/2016APV --rootfile /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/EMu_16APV_OptimalCut/leadclustertracks --var leadclustertracks --ABCD --year 2016APV --region EMu --floatB --doAll --ManualMCStats ;
wait
