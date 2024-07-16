import sys
import ROOT
import array
from math import log

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(False)
ROOT.gStyle.SetPaintTextFormat("1.3f")


class massPoint():
  def __init__(self, values):
    """ Get different variables for each of the mass points"""
    self.dictVars = {"mS": 0, "mD": 1, "T": 2, "Sig":3} 
    self.mass1    = float(values[self.dictVars["mD"]])
    self.mass2    = log(float(values[self.dictVars["T"]])/float(values[self.dictVars["mD"]]), 2)
    self.values = [float(f) for f in values]
    self.isGood = False

fil = open(sys.argv[1], "r")
binsX, binsY = eval(sys.argv[2]), eval(sys.argv[3])
bx = array.array('d', binsX)
by = array.array('d', binsY)
allPoints = []

for line in fil.readlines():
  if line[0] == "#": continue
  properValues = [a.replace(" ", "") for a in line.split(":")]
  if properValues[0] == "-99" or properValues[1] == "-99": continue
  allPoints.append( massPoint(properValues) )
  

outH = ROOT.TH2F("obs","obs", len(binsX) -1 , bx, len(binsY)-1, by)
byDown = array.array('d', [f - 0.3 for f in binsY])
outHe = ROOT.TH2F("exp","exp", len(binsX) -1 , bx, len(binsY)-1, byDown)

for p in allPoints:
  binx = outH.GetXaxis().FindBin(p.mass1)
  biny = outH.GetYaxis().FindBin(p.mass2)
  outH.SetBinContent(binx, biny, p.values[p.dictVars["Sig"]])
outH.GetZaxis().SetRangeUser(-100, 100)
c = ROOT.TCanvas("c","c", 800, 600)
#c.SetLogz(True)
c.SetRightMargin(0.20)
outH.GetXaxis().SetTitle("m_{\phi} [GeV]")
outH.GetYaxis().SetTitle("log_{2}(T/m_{\phi})")
outH.GetZaxis().SetTitle("Observed Signal Strength")
outH.SetTitle("")
#outH.SetMinimum(-0.05)
outH.Draw("textcolz")
c.SaveAs(sys.argv[4] + ".pdf")
c.SaveAs(sys.argv[4] + ".png")
