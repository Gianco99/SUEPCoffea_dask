import os


def checkFile(fi):
  #print("--->", fi)
  if os.path.isfile(fi):
    #print(fi,os.path.getsize(fi))
    if os.path.getsize(fi) > 6500: 
      return True
  return False

iJ = 0
allFiles = os.listdir("RunII")

for f in os.listdir("RunII"):
  if not("Combined" in f): continue
  if not("txt" in f): continue
  if "higgsCombine" in f: continue
  print(f)
  froot = f.replace(".txt","")
  if checkFile("RunII/higgsCombine%s.FitDiagnostics.mH120.root"%froot): continue
  print("Pass")
  iJ += 1
  fil = open("exec/job_%i.sh"%iJ, "w")
  fil.write("#!/bin/csh\n")
  fil.write("cd /eos/user/g/gdecastr/CMSSW_11_3_4/src/\n")
  fil.write("source /cvmfs/cms.cern.ch/cmsset_default.csh\n")
  fil.write("cmsenv\n")
  fil.write("cd /eos/user/g/gdecastr/SUEPCoffea_dask/Cards_Recipe/CommonBoundsABCD_ManualMC_Flattened_Int45/RunII/\n")
  fil.write("limit stacksize unlimited\n")
  fil.write("text2workspace.py %s\n"%(f))
  #if not checkFile("RunII/higgsCombine%s.AsymptoticLimits.mH120.root"%f): fil.write("combine -M AsymptoticLimits %s -n %s\n"%(f.replace("txt","root"), f.replace(".txt","")))
  #if not checkFile("RunII/higgsCombine%s.Significance.mH120.root"%f):     fil.write("combine -M Significance %s -n %s\n"%(f.replace("txt","root"), f.replace(".txt",""))) 
  if not checkFile("RunII/higgsCombine%s.FitDiagnostics.mH120.root"%f):   fil.write("combine -M FitDiagnostics --rMin -500 --rMax 500 --robustFit 1 --setParameterRanges rSR16APV=0,2:rSR16=0,2:rSR17=0,2:rSR18=0,2:rCRTT16APV=0,2:rCRTT16=0,2:rCRTT17=0,2:rCRTT18=0,2:rCRDY16APV=0,2:rCRDY16=0,2:rCRDY17=0,2:rCRDY18=0,2  --cminDefaultMinimizerStrategy 1 --setCrossingTolerance 1e-7 --saveShapes --saveWithUncertainties %s -n %s\n"%(f.replace("txt","root"), f.replace(".txt","")))
