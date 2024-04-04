#!/bin/bash

python datacardMaker.py ZH/samples_withSF_nocuts_UL18_dR_HighStats.py ZH/systs_fullcorr.py /eos/user/g/gdecastr/SUEPCoffea_dask/Cards_OptimalBounds/Blind/CRDY/2018 --rootfile /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRDY_18_OptimalBounds/leadclustertracks --var leadclustertracks --ABCD --year 2018 --region CRDY --floatB --blind --doAll ;
wait

python datacardMaker.py ZH/samples_withSF_nocuts_UL17_dR_HighStats.py ZH/systs_fullcorr.py /eos/user/g/gdecastr/SUEPCoffea_dask/Cards_OptimalBounds/Blind/CRDY/2017 --rootfile /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRDY_17_OptimalBounds/leadclustertracks --var leadclustertracks --ABCD --year 2017 --region CRDY --floatB --blind --doAll ;
wait

python datacardMaker.py ZH/samples_withSF_nocuts_UL16_dR_HighStats.py ZH/systs_fullcorr.py /eos/user/g/gdecastr/SUEPCoffea_dask/Cards_OptimalBounds/Blind/CRDY/2016 --rootfile /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRDY_16_OptimalBounds/leadclustertracks --var leadclustertracks --ABCD --year 2016 --region CRDY --floatB --blind --doAll ;
wait

python datacardMaker.py ZH/samples_withSF_nocuts_UL16APV_dR_HighStats.py ZH/systs_fullcorr.py /eos/user/g/gdecastr/SUEPCoffea_dask/Cards_OptimalBounds/Blind/CRDY/2016APV --rootfile /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRDY_16APV_OptimalBounds/leadclustertracks --var leadclustertracks --ABCD --year 2016APV --region CRDY --floatB --blind --doAll ;
wait