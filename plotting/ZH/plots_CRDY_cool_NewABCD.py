import ROOT
import numpy as np
import pandas as pd

def compute_deltaR(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = np.arccos(np.cos(phi1 - phi2))
    return np.sqrt(deta ** 2 + dphi ** 2)

def leadDeltaR(x):
  x['deltaR_leadjet'] = compute_deltaR(x['leadjet_eta'], x['leadjet_phi'], x['leadcluster_eta'], x['leadcluster_phi'])
  return x['deltaR_leadjet']

def cut(x):
  return (x["njets"] >= 0) & (x["Z_m"] > 120) & (x["Z_pt"] >= 25) & (x["nBLoose"] == 0) & (x["leadcluster_pt"] >= 60) & (compute_deltaR(x['leadjet_eta'], x['leadjet_phi'], x['leadcluster_eta'], x['leadcluster_phi']) <= 1.5)

def byRegion(x):
  nTrackReg = (x["njets"] < 0)
  ptReg1 = (x["leadjet_pt"] < 135)*1 
  ptReg2 = (x["leadjet_pt"] < 220)*1
  nTrackReg = (x["leadcluster_ntracks"] >= 14) + np.clip(np.where((x["leadcluster_ntracks"]-15.99)/5 > 0 , (x["leadcluster_ntracks"]-15.99)/5, 0), 0, 6)
  #print(ptReg1[:10], ptReg2[:10], nTrackReg[:10], x["leadjet_pt"][:10], x["leadcluster_ntracks"][:10])
  ret = (8*ptReg1) + (8*ptReg2) + nTrackReg
  #print(ret) 
  return ret 
plots = {
  "bins": {
             "name"     : "bins", 
             "bins"     : ["uniform", 24, 0, 24],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (byRegion(x), y*cut(x)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 5e10,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "bins",
             "xlabel"   : "N_{Tracks}^{SUEP}",
             "legendPosition": [0.12, 0.65, 0.90, 0.86],
             "xbinlabels": ["0-14","14-21","21-26","26-31","31-36", "36-41","41-46",">46", "0-14","14-21","21-26","26-31","31-36", "36-41","41-46",">46", "0-14","14-21","21-26","26-31","31-36", "36-41","41-46",">46"],
             "vars"     : ["njets"],
             "lines"    : [ [1,2e-1,1,1e7,7,3,48], [2,2e-1,2,1e7,7,3,48], [8,2e-1,8,1e8,9,5,ROOT.kBlack], [9,2e-1,9,1e7,7,3,48], [10,2e-1,10,1e7,7,3,48],  [16,2e-1,16,1e8,9,5,ROOT.kBlack],[17,2e-1,17,1e7,7,3,48], [18,2e-1,18,1e7,7,3,48]],
             "text"     : [["p_{T}^{jet1} > 220 GeV", [2.4,3e7], 0.05, ROOT.kBlack], ["135 GeV < p_{T}^{jet1} #leq 220 GeV", [8.4,3e7], 0.045, ROOT.kBlack],["p_{T}^{jet1} #leq 135 GeV", [18.4,3e7], 0.05, ROOT.kBlack], ["E2", [0.25,5e6], 0.03, 48], ["E1", [1.25,5e6], 0.03, 48], ["B2", [3.25,5e6], 0.03, 48], ["D2", [8.25,5e6], 0.03, 48], ["D1", [9.25,5e6], 0.03, 48], ["B1", [13.25,5e6], 0.03, 48], ["C2", [16.25,5e6], 0.03, 48], ["C1", [17.25,5e6], 0.03, 48], ["A", [21.25,5e6], 0.03, 48]],
             "rlines"   : [[1,0.0,1,2.0,7,3,48], [2,0.0,2,2.0,7,3,48], [8,0.0,8,2.0,9,5,ROOT.kBlack],[9,0.0,9,2.0,7,3,48], [10,0.0,10,2.0,7,3,48], [16,0.0,16,2.0,9,5,ROOT.kBlack],[17,0.0,17,2.0,7,3,48], [18,0.0,18,2.0,7,3,48]],
  },
}

 
