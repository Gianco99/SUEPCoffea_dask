import ROOT


def cut(x):
  return (x["njets"] >= 0)  

plots = {
  "bins": {
             "name"     : "bins",
             "bins"     : ["uniform", 30, 0, 30],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["njets"]>=0, y*cut(x)),
             "logY"     : True,
             "normalize": False,
             "maxY"     : 5e10,
             "minY"     : 1e0,
             "ratiomaxY": 5.,
             "ratiominY": 0.,
             "plotname" : "bins",
             "xlabel"   : "N_{Tracks}^{SUEP}",
             "legendPosition": [0.12, 0.65, 0.90, 0.86],
             "xbinlabels": ["0-10","10-20","20-25","25-30","30-35", "35-40","40-45",">45", "0-10","10-20","20-25","25-30","30-35", "35-40","40-45",">45","0-10","10-20","20-25","25-30","30-35", "35-40","40-45",">45"],
             "vars"     : ["njets"],
             "lines"    : [ [1,2e-1,1,1e7,7,3,48], [2,2e-1,2,1e7,7,3,48], [8,2e-1,8,1e8,9,5,ROOT.kBlack], [9,2e-1,9,1e7,7,3,48], [10,2e-1,10,1e7,7,3,48],  [16,2e-1,16,1e8,9,5,ROOT.kBlack],[17,2e-1,17,1e7,7,3,48], [18,2e-1,18,1e7,7,3,48]],
             "text"     : [["p_{T}^{jet1} > 250 GeV", [1.92,3e7], 0.05, ROOT.kBlack], ["100 GeV < p_{T}^{jet1} #leq 250 GeV", [8.92,3e7], 0.045, ROOT.kBlack],["p_{T}^{jet1} #leq 100 GeV", [17.92,3e7], 0.05, ROOT.kBlack], ["E2", [0.25,5e6], 0.03, 48], ["E1", [1.25,5e6], 0.03, 48], ["B2", [4.25,5e6], 0.03, 48], ["D2", [8.25,5e6], 0.03, 48], ["D1", [9.25,5e6], 0.03, 48], ["B1", [12.25,5e6], 0.03, 48], ["C2", [16.25,5e6], 0.03, 48], ["C1", [17.25,5e6], 0.03, 48], ["A", [20.25,5e6], 0.03, 48]],
             "rlines"   : [[1,0.0,1,8.0,7,3,48], [2,0.0,2,8.0,7,3,48], [8,0.0,8,8.0,9,5,ROOT.kBlack],[9,0.0,9,8.0,7,3,48], [10,0.0,10,8.0,7,3,48], [16,0.0,16,8.0,9,5,ROOT.kBlack],[17,0.0,17,8.0,7,3,48], [18,0.0,18,8.0,7,3,48]],
  },
}

 