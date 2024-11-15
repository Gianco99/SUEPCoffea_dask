#!/bin/bash

#Default
python plotter_vh.py ZH/samples_withSF_nocuts_UL18_dR_HighStats.py ZH/plots_forCards_withDeltaR.txt -l 59.9 --systFile ZH/systs_fullcorr_MC.py --toSave /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_18_Nominal_Fixed  --batchsize 100 --jobname  /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_18_Jobs_Nominal_Fixed --plotAll --resubmit --queue workday ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL17_dR_HighStats.py ZH/plots_forCards_withDeltaR.txt -l 41.6 --systFile ZH/systs_fullcorr_MC.py --toSave /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_17_Nominal_Fixed  --batchsize 100 --jobname  /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_17_Jobs_Nominal_Fixed --plotAll --resubmit --queue workday ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL16_dR_HighStats.py ZH/plots_forCards_withDeltaR.txt -l 16.4 --systFile ZH/systs_fullcorr_MC.py --toSave /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_16_Nominal_Fixed  --batchsize 100 --jobname  /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_16_Jobs_Nominal_Fixed --plotAll --resubmit --queue workday ;
wait

python plotter_vh.py ZH/samples_withSF_nocuts_UL16APV_dR_HighStats.py ZH/plots_forCards_withDeltaR.txt -l 19.9 --systFile ZH/systs_fullcorr_MC.py --toSave /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_16APV_Nominal_Fixed  --batchsize 100 --jobname  /eos/cms/store/group/phys_exotica/SUEPs/ZH_Histos/SR_16APV_Jobs_Nominal_Fixed --plotAll --resubmit --queue workday ;
wait