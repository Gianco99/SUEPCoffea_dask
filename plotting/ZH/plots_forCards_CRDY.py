import ROOT


def cut(x):
  return (x["njets"] >= 0) & (x["Z_m"] > 120) & (x["Z_pt"] >= 25) & (x["nBLoose"] == 0) & (x["leadcluster_pt"] >= 60)

plots = {
  "leadjet_pt": {
             "name"     : "leadjet_pt",
             "bins"     : ["uniform", 50, 0, 500],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadjet_pt"], y*cut(x)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadjet_pt",
             "xlabel"   : "p_{T}^{jet1}",
             "vars"     : ["leadjet_pt"]
  },
  "leadclustertracks": {
             "name"     : "leadclustertracks",
             "bins"     : ["uniform", 25, 0, 100],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclustertracks",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },
 "leadclustertracks_onecluster": {
             "name"     : "leadclustertracks_onecluster",
             "bins"     : ["uniform", 8, 20, 60],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)*(x["leadjet_pt"] < 100)*(x["leadcluster_ntracks"] >= 20)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclustertracks_onecluster",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },
 "leadclustertracks_B1": {
             "name"     : "leadclustertracks_B1",
             "bins"     : ["uniform", 8, 20, 60],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)*(x["leadjet_pt"] > 100)*(x["leadjet_pt"] < 250)*(x["leadcluster_ntracks"] >= 20)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclustertracks_B1",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },
 "leadclustertracks_B2": {
             "name"     : "leadclustertracks_B2",
             "bins"     : ["uniform", 8, 20, 60],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)*(x["leadjet_pt"] > 250)*(x["leadcluster_ntracks"] >= 20)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclustertracks_B2",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },

 "leadclustertracks_C1": {
             "name"     : "leadclustertracks_C1",
             "bins"     : ["uniform", 1, 10, 20],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)*(x["leadjet_pt"] < 100)*(x["leadcluster_ntracks"] < 20)* (x["leadcluster_ntracks"] >= 10)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclustertracks_C1",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },
 "leadclustertracks_C2": {
             "name"     : "leadclustertracks_C2",
             "bins"     : ["uniform", 1, 0, 10],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)*(x["leadjet_pt"] < 100)*(x["leadcluster_ntracks"] < 10)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclustertracks_C2",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },

 "leadclustertracks_D1": {
             "name"     : "leadclustertracks_D1",
             "bins"     : ["uniform", 1, 10, 20],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)*(x["leadjet_pt"] < 250)*(x["leadjet_pt"] > 100)*(x["leadcluster_ntracks"] < 20)* (x["leadcluster_ntracks"] >= 10)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclustertracks_D1",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },
 "leadclustertracks_D2": {
             "name"     : "leadclustertracks_D2",
             "bins"     : ["uniform", 1, 0, 10],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)*(x["leadjet_pt"] < 250)*(x["leadjet_pt"] > 100)*(x["leadcluster_ntracks"] < 10)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclustertracks_D2",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },

 "leadclustertracks_E1": {
             "name"     : "leadclustertracks_E1",
             "bins"     : ["uniform", 1, 10, 20],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)*(x["leadjet_pt"] > 250)*(x["leadcluster_ntracks"] < 20)* (x["leadcluster_ntracks"] >= 10)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclustertracks_E1",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },
 "leadclustertracks_E2": {
             "name"     : "leadclustertracks_E2",
             "bins"     : ["uniform", 1, 0, 10],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)*(x["leadjet_pt"] > 250)*(x["leadcluster_ntracks"] < 10)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 1e9,
             "minY"     : 1e0,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclustertracks_E2",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },
}

 
