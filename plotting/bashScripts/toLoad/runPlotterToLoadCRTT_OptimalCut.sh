#!/bin/bash

# Default
python plotter_vh.py ZH/samples_withSF_nocuts_UL18_dR_HighStats.py ZH/plots_forCardsCRTT_dR_OptimalCut.txt -l 59.9 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/CRTT_18_OptimalBounds  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRTT_18_OptimalBounds --plotAll --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL17_dR_HighStats.py ZH/plots_forCardsCRTT_dR_OptimalCut.txt -l 41.6 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/CRTT_17_OptimalBounds  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRTT_17_OptimalBounds --plotAll --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL16_dR_HighStats.py ZH/plots_forCardsCRTT_dR_OptimalCut.txt -l 16.4 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/CRTT_16_OptimalBounds  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRTT_16_OptimalBounds --plotAll --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL16APV_dR_HighStats.py ZH/plots_forCardsCRTT_dR_OptimalCut.txt -l 19.9 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/CRTT_16APV_OptimalBounds  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRTT_16APV_OptimalBounds --plotAll --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_RunII_dR_HighStats.py ZH/plots_forCardsCRTT_dR_OptimalCut.txt -l 137.8 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/CRTT_{YEAR}_OptimalBounds  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRTT_RunII_OptimalBounds --plotAll --strict-order ;
wait