#!/bin/bash

# Default
python plotter_vh.py ZH/samples_withSF_nocuts_UL18_dR.py ZH/plots_forCards_withDeltaR_ZMassPm5.txt -l 59.9 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_18_ZMassPm5  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/SR_18_ZMassPm5 --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL17_dR.py ZH/plots_forCards_withDeltaR_ZMassPm5.txt -l 41.6 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_17_ZMassPm5  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/SR_17_ZMassPm5 --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL16_dR.py ZH/plots_forCards_withDeltaR_ZMassPm5.txt -l 16.4 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_16_ZMassPm5  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/SR_16_ZMassPm5 --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL16APV_dR.py ZH/plots_forCards_withDeltaR_ZMassPm5.txt -l 19.9 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_16APV_ZMassPm5 --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/SR_16APV_ZMassPm5 --strict-order ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_RunII_dR.py ZH/plots_forCards_withDeltaR_ZMassPm5.txt -l 137.8 --systFile ZH/systs_fullcorr_MC.py --toLoad /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_{YEAR}_ZMassPm5  --plotdir  /eos/user/g/gdecastr/SUEPCoffea_dask/Plots/SR_RunII_ZMassPm5 --strict-order ;
wait