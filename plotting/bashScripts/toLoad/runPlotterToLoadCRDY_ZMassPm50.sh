#!/bin/bash

# Default
python plotter_vh.py ZH/samples_withSF_nocuts_UL18_dR.py ZH/plots_forCards_withDeltaR_CRDY_ZMassPm50.txt -l 59.9 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/CRDY_18_ZMassPm50  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRDY_18_ZMassPm50 --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL17_dR.py ZH/plots_forCards_withDeltaR_CRDY_ZMassPm50.txt -l 41.6 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/CRDY_17_ZMassPm50  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRDY_17_ZMassPm50 --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL16_dR.py ZH/plots_forCards_withDeltaR_CRDY_ZMassPm50.txt -l 16.4 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/CRDY_16_ZMassPm50  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRDY_16_ZMassPm50 --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL16APV_dR.py ZH/plots_forCards_withDeltaR_CRDY_ZMassPm50.txt -l 19.9 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/CRDY_16APV_ZMassPm50  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRDY_16APV_ZMassPm50 --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_RunII_dR.py ZH/plots_forCards_withDeltaR_CRDY_ZMassPm50.txt -l 137.8 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/CRDY_{YEAR}_ZMassPm50  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/CRDY_RunII_ZMassPm50 --strict-order ;
wait