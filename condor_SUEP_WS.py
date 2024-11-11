import os
import json
import argparse

#Import coffea specific features
from coffea.processor import run_uproot_job, futures_executor


#Begin argparse
parser = argparse.ArgumentParser("")
parser.add_argument('--isMC', type=int, default=1, help="")
parser.add_argument('--jobNum', type=int, default=1, help="")
parser.add_argument('--era', type=str, default="2018", help="")
parser.add_argument('--doSyst', type=int, default=1, help="")
parser.add_argument('--infile', type=str, default=None, help="")
parser.add_argument('--dataset', type=str, default="X", help="")
parser.add_argument('--nevt', type=str, default=-1, help="")
parser.add_argument('--analyzer', type=str, default="GluGlu", help="")
parser.add_argument('--outputdir', type=str, default=None, help="")
parser.add_argument('--chunksize', type=int, default=500000, help="")
parser.add_argument('--test', type=bool, default=False, help="")
parser.add_argument('--isDY', action="store_true", default=False, help="Activate for the DY Zpt=0 fix for UL")
parser.add_argument('--SR', action="store_true", default=False, help="Activate to only save yields at SR level to save space")
parser.add_argument('--fileName', type=str, default='file', help="File extension for Jet Resolution CSVs")
options = parser.parse_args()

out_dir = options.outputdir if options.outputdir else os.getcwd()

print("isDY", options.isDY)
## Select analyzer
modules_era = []
if options.analyzer == "GluGlu":
  from workflows.SUEP_coffea import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , flag=False, output_location=out_dir))
elif options.analyzer == "ZH_simple":
  from workflows.SUEP_coffea_ZH_simple import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_bootstrap":
  from workflows.SUEP_coffea_ZH_simple_bootstrapping import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst":
  from workflows.SUEP_coffea_ZH_simple import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_fortests":
  from workflows.SUEP_coffea_ZH_simple_fortests import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_OF":
  from workflows.SUEP_coffea_ZH_simple import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=True))
elif options.analyzer == "ZH_trackID":
  from workflows.SUEP_coffea_ZH_trackID import * 
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , flag=False, output_location=out_dir))
elif options.analyzer == "ZH_trackProps":
  from workflows.SUEP_coffea_ZH_trackProps import * 
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , flag=False, output_location=out_dir))
elif options.analyzer == "ZH_gen":
  from workflows.SUEP_coffea_ZH_onlygenZpt import * 
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , flag=False, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_leptonID":
  from workflows.SUEP_coffea_ZH_leptonID import * 
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=True, isDY = options.isDY))
elif options.analyzer == "ZH_btagEff":
  from workflows.SUEP_coffea_ZH_bTagEff import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=True, isDY = options.isDY))
elif options.analyzer == "MuonID":
  from workflows.SUEP_coffea_ZH_checkMuonID import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ntracksplit":
  from workflows.SUEP_coffea_ZH_simple_ntracksplit import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "checkJetID":
  from workflows.SUEP_coffea_ZH_checkJetID import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_4p0":
  from workflows.SUEP_coffea_ZH_simple_4p0 import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_2p0":
  from workflows.SUEP_coffea_ZH_simple_2p0 import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_1p2":
  from workflows.SUEP_coffea_ZH_simple_1p2 import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_p8":
  from workflows.SUEP_coffea_ZH_simple_p8 import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_p4":
  from workflows.SUEP_coffea_ZH_simple_p4 import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_NoSR":
  from workflows.SUEP_coffea_ZH_simple_NoSR import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_NewID":
  from workflows.SUEP_coffea_ZH_simple_NewID import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_Cambridge":
  from workflows.SUEP_coffea_ZH_simple_Cambridge import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_Cambridge_4p0":
  from workflows.SUEP_coffea_ZH_simple_Cambridge_4p0 import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_Cambridge_2p0":
  from workflows.SUEP_coffea_ZH_simple_Cambridge_2p0 import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_Cambridge_1p2":
  from workflows.SUEP_coffea_ZH_simple_Cambridge_1p2 import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_Cambridge_p8":
  from workflows.SUEP_coffea_ZH_simple_Cambridge_p8 import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_Cambridge_p4":
  from workflows.SUEP_coffea_ZH_simple_Cambridge_p4 import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_ThinZMass":
  from workflows.SUEP_coffea_ZH_simple_ThinZMass import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_NewID_NoSR":
  from workflows.SUEP_coffea_ZH_simple_NewID_NoSR import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_withsyst_SideBandA":
  from workflows.SUEP_coffea_ZH_simple_SideBandA import *
  modules_era.append(SUEP_cluster(isMC=False, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=False))
elif options.analyzer == "ZH_simple_withsyst_SideBandB1":
  from workflows.SUEP_coffea_ZH_simple_SideBandB1 import *
  modules_era.append(SUEP_cluster(isMC=False, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=False))
elif options.analyzer == "ZH_simple_withsyst_MatchingJets":
  from workflows.SUEP_coffea_ZH_MatchingJets import *
  modules_era.append(SUEP_cluster(isMC=False, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=False))
elif options.analyzer == "ZH_simple_withsyst_Neutrals":
  from workflows.SUEP_coffea_ZH_simple_Neutrals import *
  modules_era.append(SUEP_cluster(isMC=False, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=False))
elif options.analyzer == "ZH_simple_twolepton":
  from workflows.SUEP_coffea_ZH_simple_twolepton import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_onecluster":
  from workflows.SUEP_coffea_ZH_simple_onecluster import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_onecluster_isDYTrue":
  from workflows.SUEP_coffea_ZH_simple_onecluster import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=options.SR, output_location=out_dir, doOF=False, isDY=True))
elif options.analyzer == "ZH_simple_Leptons_wIso":
  from workflows.SUEP_coffea_ZH_simple_Leptons_wIso import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=False, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_Leptons_NoIso":
  from workflows.SUEP_coffea_ZH_simple_Leptons_NoIso import *
  modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample=options.dataset, weight_syst='' , SRonly=False, output_location=out_dir, doOF=False, isDY=options.isDY))
elif options.analyzer == "ZH_simple_RecoJets":
  from workflows.SUEP_coffea_ZH_simple_RecoJets import *
  modules_era.append(SUEP_cluster(fileName=options.fileName, isMC=True, era=int(options.era), do_syst=0,  syst_var='', sample='SUEP', weight_syst='' , SRonly=False, output_location=out_dir, doOF=False, isDY=False))
elif options.analyzer == "ZH_simple_VBF":
  from workflows.SUEP_coffea_ZH_simple_VBF import *
  modules_era.append(SUEP_cluster(fileName=options.fileName, isMC=options.isMC, era=int(options.era), do_syst=0,  syst_var='', sample='SUEP', weight_syst='' , SRonly=False, output_location=out_dir, doOF=False, isDY=False))
for instance in modules_era:
    output = run_uproot_job(
        {instance.sample: [options.infile]},
        treename='Events',
        processor_instance=instance,
        executor=futures_executor,
        executor_args={'workers': 1,
                       'schema': processor.NanoAODSchema,
                       'xrootdtimeout': 10,
        },
        chunksize = 100 if options.test else options.chunksize,
        maxchunks = 1 if options.test else None,
    )
