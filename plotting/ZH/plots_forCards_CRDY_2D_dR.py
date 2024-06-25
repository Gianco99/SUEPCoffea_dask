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

plots = {
}

var = {
"N_C" : ["leadcluster_ntracks", "N_{tracks}^{SUEP}", 100, 0, 100],
"jet1pt":["leadjet_pt", "p_{T}^{jet1} [GeV]", 200, 0, 1000],
}

v1 = "N_C"
v2 = "jet1pt"
name = "2D_%s_%s"%(v1,v2)
plots[name] = {}
plots[name]["name"]    = name
plots[name]["bins"]    = ["2Duniform", var[v1][2], var[v1][3], var[v1][4], var[v2][2], var[v2][3], var[v2][4]]
plots[name]["channel"] = "onecluster"
plots[name]["value"]   = lambda x, y, v1=v1, v2=v2: (x[var[v1][0]], x[var[v2][0]], y*cut(x))
plots[name]["logY"     ]= True
plots[name]["normalize"]= False
plots[name]["maxY"]     = 1e9
plots[name]["minY"]     = 1e0
plots[name]["ratiomaxY"]= 2.
plots[name]["ratiominY"]= 0.
plots[name]["plotname"] = name
plots[name]["xlabel"]   = var[v1][1]
plots[name]["ylabel"]   = var[v2][1]
plots[name]["vars"]     = [var[v1][0], var[v2][0]]
plots[name]["mode"]     = "colz"


