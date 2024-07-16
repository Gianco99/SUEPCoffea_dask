import ROOT
import imp
import copy
import re 
import os

ROOT.gErrorIgnoreLevel = ROOT.kError

class datacardMaker(object):
  def __init__(self, samples, systs, output, options, channel = "bin1"):
    self.samples = samples
    self.systs   = systs
    self.output  = output
    self.options = options
    self.isCC    = False
    self.isShape = False
    self.signals = []
    self.backgr  = []
    self.data    = []
    self.channel = channel
  def createDatacards(self):
    self.collectYields()
    for s in self.samples:
      if "isSig" in self.samples[s]:
        if self.samples[s]["isSig"]:
          if self.options.thispoint:
            if s != self.options.thispoint: continue
          print("Create card for signal process %s..."%(s))
          self.createDatacard(s)
    return self.nbins
  def createDatacard(self, s):
    if not(self.options.ABCD):
      processes = [s] + self.backgr
      bins      = ["1"]
      empty = ["-"]*len(processes)
      newcard = open(self.output + "/" + self.options.region +  self.samples[s]["name"] + ".txt","w")
      newcard.write("imax *\njmax *\nkmax *\n")

      if self.isShape:
        newcard.write("shapes * * ./" + options.region + self.samples[s]["name"] + "_shapes.root $PROCESS $PROCESS$SYSTEMATIC\n")
      newcard.write("bin %s\n"%self.channel)
      newcard.write("observation -1\n")
      newcard.write("bin " + " ".join(["%s"%self.channel for i in range(len(self.backgr)+1)])+ "\n")
      newcard.write("process " + s + " " + " ".join(self.backgr)+"\n")
      newcard.write("process " + " ".join([str(i) for i in range(len(self.backgr)+1)])+"\n")
      newcard.write("rate " + " ".join(["-1" for i in range(len(self.backgr)+1)])+"\n")

      for syst in self.systs: # Now the fun begins:
        if self.systs[syst]["type"] == "lnN" or self.systs[syst]["type"] == "gmN":
          onlychannel = None
          if "onlychannel" in self.systs[syst]: 
            onlychannel = self.systs[syst]["onlychannel"]
          if onlychannel:
            if self.channel != onlychannel:
              print("Will skip %s for channel %s"%(syst, self.channel)) 
              continue
          size = "-"
          name = syst
          for corrScheme in self.systs[syst]["corrs"]:
            if options.year in self.systs[syst]["corrs"][corrScheme]:
              name = syst + "_" + corrScheme
              for iyear, yearcorr in enumerate(self.systs[syst]["corrs"][corrScheme]):
                if options.year == yearcorr:
                  if size == "-": 
                    if type(self.systs[syst]["size"]) == type({"a":1}):
                      size = "%1.3f"%self.systs[syst]["size"][corrScheme][iyear] 
                    else:
                      size = "%1.3f"%self.systs[syst]["size"]
          if len(self.systs[syst]["corrs"]) == 1: 
            name = syst

          perbin = False
          if "perbin" in self.systs[syst]: 
            perbin = self.systs[syst]["perbin"]
          if not(perbin):
            #print(processes)
            effect = [size if any([re.match(princfg, princard)  for princfg in self.systs[syst]["processes"]]) else "-" for princard in processes]
            #print(size, effect, processes,  self.systs[syst]["processes"])
            if effect == empty:
              print("......lnN/gmN Systematic %s has no effect, will skip it"%(name))
            else:
              newcard.write("%s %s "%(name, self.systs[syst]["type"]) + " ".join(effect)+ " \n")
          else:
            basename = name
            for ibin in range(1, self.nbins+1):
              name = basename + "_bin"+str(ibin)
              #print(name, len(processes), len(bins))
              effect = [size if any([re.match(princfg, processes[ientry]) and (("%sbin%i"%(self.channel,ibin)) == bins[ientry])  for princfg in self.systs[syst]["processes"]]) else "-" for ientry in range(len(processes))]
              if effect == empty:
                print("......lnN/gmN Systematic %s has no effect, will skip it"%(name))
                continue
              else:
                newcard.write("%s %s "%(name, self.systs[syst]["type"]) + " ".join(effect)+ " \n")

        elif self.systs[syst]["type"] == "shape":
          onlychannel = None
          name = syst
          #print("Shape", syst)
          if hasattr(self.systs[syst], "onlychannel"):
            onlychannel = self.systs[syst]["onlychannel"]
          if onlychannel:
            if self.channel != onlychannel:
              continue
          size = "-"
          for corrScheme in self.systs[syst]["corrs"]:
            if options.year in self.systs[syst]["corrs"][corrScheme]:
              name = syst + "_" + corrScheme
              size = "1"
          if len(self.systs[syst]["corrs"]) == 1: 
            name = syst
          effect = [size if any([re.match(princfg, princard)  for princfg in self.systs[syst]["processes"]]) else "-" for princard in processes]
          #print("Effect", effect)
          if effect == empty:
            print("......Shape Systematic %s has no effect, will skip it"%(name))
            continue
          newcard.write("%s %s "%(name, self.systs[syst]["type"]) + " ".join(effect)+ " \n")
          #print(size, effect, processes,  self.systs[syst]["processes"])
          for pr in processes:
            #print(pr, princfg,pr)
            if any([re.match(princfg, pr)  for princfg in self.systs[syst]["processes"]]): 
              print("Match",self.systs[syst]["match"].replace("$PROCESS", self.samples[pr]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", self.channel) + self.systs[syst]["up"], self.systs[syst]["match"].replace("$PROCESS", self.samples[pr]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", self.channel) + self.systs[syst]["down"])
              newup = self.tf.Get(self.systs[syst]["match"].replace("$PROCESS", self.samples[pr]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", self.channel) + self.systs[syst]["up"])
              newdn = self.tf.Get(self.systs[syst]["match"].replace("$PROCESS", self.samples[pr]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", self.channel) + self.systs[syst]["down"])
              self.th1s[pr + syst + "Up"] = copy.deepcopy(newup.Clone(pr + name + "Up"))
              self.th1s[pr + syst + "Down"] = copy.deepcopy(newdn.Clone(pr + name + "Down"))
              #if (self.th1s[pr + syst + "Up"].Integral() < 0.0001) or (self.th1s[pr + syst + "Down"].Integral() < 0.0001):
              for ibin in range(1, self.th1s[pr + syst + "Down"].GetNbinsX()+1):
                  if self.th1s[pr + syst + "Down"].GetBinContent(ibin) < 0.0001:  self.th1s[pr + syst + "Down"].SetBinContent(ibin,0.0001)
                  if self.th1s[pr + syst + "Up"].GetBinContent(ibin) < 0.0001:  self.th1s[pr + syst + "Up"].SetBinContent(ibin,0.0001)

      if self.isShape: 
        newcard.write("* autoMCStats 0\n")
        rout = ROOT.TFile(self.output + "/" + options.region + self.samples[s]["name"] + "_shapes.root", "RECREATE")
        rout.cd()
        #for b in self.backgr:
        #  self.th1s[b].Write()
        #self.th1s["data_obs"].Write()
        #self.th1s[s].Write()
        for h in self.th1s:
          self.th1s[h].Write()
        rout.Close()
      newcard.close()
    else:
      if os.path.isfile(self.output + "/" + options.region + "%s_"%self.channel + self.samples[s]["name"] + "_shapes.root"):
        if os.path.getsize(self.output + "/" + options.region + "%s_"%self.channel + self.samples[s]["name"] + "_shapes.root") < 100:
          os.system("rm %s"%(self.output + "/" + options.region + "%s_"%self.channel + self.samples[s]["name"] + "_shapes.root"))
        else:
          print("Skip %s"%(self.output + "/" + options.region + "%s_"%self.channel + self.samples[s]["name"] + "_shapes.root"))
          return


      newcard = open(self.output + "/" + self.options.region + "%s"%self.channel + "_" + self.samples[s]["name"] + ".txt","w")
      newcard.write("imax *\njmax *\nkmax *\n")
      if self.isShape:
        newcard.write("shapes * * ./" + options.region + "%s"%self.channel + "_" + self.samples[s]["name"] + "_shapes.root $PROCESS_$CHANNEL $PROCESS_$CHANNEL_$SYSTEMATIC \n")
      newcard.write("bin " + " ".join(["%sbin%i"%(self.channel, i) for i in range(1,self.nbins+1)])  + " \n")
      newcard.write("observation " + " ".join(["-1"  for ibin in range(1, self.nbins+1)]) + " \n")
      newcard.write("bin " + " ".join([ " ".join(["%sbin%i"%(self.channel, j) for i in range(2)]) for j in range(1,self.nbins+1)])+" \n")
      newcard.write("process " + " ".join([s + " total_background" for i in range(self.nbins)])+"\n")
      newcard.write("process " + " ".join(["0 1" for i in range(self.nbins)])+"\n")
      newcard.write("rate " + " ".join(["-1 -1" for ibin in range(1, self.nbins+1)])+" \n")
      processes = []
      bins      = []
      for ibin in range(1,self.nbins+1):
        processes.append(s)
        processes.append("total_background")
        bins.append("%sbin%i"%(self.channel, ibin))
        bins.append("%sbin%i"%(self.channel, ibin))

      empty = ["-"]*len(processes)
      for syst in self.systs: # Now the fun begins:
        #print(syst)
        if self.systs[syst]["type"] == "lnN" or self.systs[syst]["type"] == "gmN":
          onlychannel = None
          if "onlychannel" in self.systs[syst]: 
            onlychannel = self.systs[syst]["onlychannel"]
          if onlychannel:
            if self.channel != onlychannel:
              print("Will skip %s for channel %s"%(syst, self.channel)) 
              continue
          size = "-"
          name = syst
          for corrScheme in self.systs[syst]["corrs"]:
            if options.year in self.systs[syst]["corrs"][corrScheme]:
              name = syst + "_" + corrScheme
              for iyear, yearcorr in enumerate(self.systs[syst]["corrs"][corrScheme]):
                if options.year == yearcorr:
                  if size == "-": 
                    if type(self.systs[syst]["size"]) == type({"a":1}):
                      size = "%1.3f"%self.systs[syst]["size"][corrScheme][iyear] 
                    else:
                      size = "%1.3f"%self.systs[syst]["size"]
          if len(self.systs[syst]["corrs"]) == 1: 
            name = syst

          perbin = False
          if "perbin" in self.systs[syst]: 
            perbin = self.systs[syst]["perbin"]
          if not(perbin):
            effect = [size if any([re.match(princfg, princard)  for princfg in self.systs[syst]["processes"]]) else "-" for princard in processes]
            #print(size, effect, processes,  self.systs[syst]["processes"])
            if effect == empty:
              print("......lnN/gmN Systematic %s has no effect, will skip it"%(name))
            else:
              newcard.write("%s %s "%(name, self.systs[syst]["type"]) + " ".join(effect)+ " \n")
          else:
            basename = name
            for ibin in range(1, self.nbins+1):
              name = basename + "_bin"+str(ibin)
              effect = [size if any([re.match(princfg, processes[ientry]) and (("%sbin%i"%(self.channel,ibin)) == bins[ientry])  for princfg in self.systs[syst]["processes"]]) else "-" for ientry in range(len(processes))]
              if effect == empty:
                print("......lnN/gmN Systematic %s has no effect, will skip it"%(name))
                continue
              else:
                newcard.write("%s %s "%(name, self.systs[syst]["type"]) + " ".join(effect)+ " \n")

        elif self.systs[syst]["type"] == "shape":
          onlychannel = None
          name = syst
          if hasattr(self.systs[syst], "onlychannel"):
            onlychannel = self.systs[syst]["onlychannel"]
          if onlychannel:
            if self.channel != onlychannel:
              continue
          size = "-"
          for corrScheme in self.systs[syst]["corrs"]:
            if options.year in self.systs[syst]["corrs"][corrScheme]:
              name = syst + "_" + corrScheme
              size = "1"
          if len(self.systs[syst]["corrs"]) == 1: 
            name = syst
          effect = [size if any([re.match(princfg, princard)  for princfg in self.systs[syst]["processes"]]) else "-" for princard in processes]
          if effect == empty:
            print("......Shape Systematic %s has no effect, will skip it"%(name))
            continue
          newcard.write("%s %s "%(name, self.systs[syst]["type"]) + " ".join(effect)+ " \n")
          for ientry, entry in enumerate(effect):
            if entry == "-": 
              continue #No effect

            ss = processes[ientry]
            factor = 1
            if ss in self.signals: factor = options.scaleS
            # Else we have to get the shapes and save them

            newup = self.tf.Get((self.systs[syst]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", self.channel) if not("total_background" in ss) else ("total_background" + "_" + syst)) + self.systs[syst]["up"])
            if self.systs[syst]["down"] == "":
              newdn = self.tf.Get(self.systs["yields"]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("[CHANNEL]", self.channel) if not("total_background" in ss) else "total_background")
            else:
              newdn = self.tf.Get((self.systs[syst]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", self.channel) if not("total_background" in ss) else ("total_background" + "_" + syst)) + self.systs[syst]["down"])
            #print((self.systs[syst]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", self.channel) if not("total_background" in ss) else ("total_background" + "_" + syst )) + self.systs[syst]["up"], (self.systs[syst]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", self.channel) if not("total_background" in ss) else ("total_background" + "_" + syst)) + self.systs[syst]["down"])
            #newdn = self.tf.Get(self.systs[syst]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", self.channel) + self.systs[syst]["down"]) #if self.systs[syst]["down"] != "" else self.th1s_perbin[ibin][s]
            #print(ss, syst,  self.th1s_perbin[ibin][ss], newdn.Print("all"))
            if "flatten" in self.systs[syst]:
              for ibin in range(1, self.nbins+1):
                subch = self.channel + "_bin%i"%ibin
                groupsFlat = self.systs[syst]["flatten"]
                what = {}
                whatWithBins = None
                for group in groupsFlat:
                  if subch in group: whatWithBins = group
                if not(whatWithBins): raise("Flatten uncertainty %s undefined for channel %s"%(syst, self.channel))
                for wBins in whatWithBins:
                  ch, bininch = wBins.split("_")
                  bininch = int(bininch.replace("bin",""))
                  if not(ch in what): what[ch] = [bininch]
                  else: what[ch].append(bininch)
                if not(whatWithBins): raise("Flatten uncertainty %s undefined for channel %s"%(syst, self.channel))
                #print("For systematic %s on channel %s will combine information from:"%(syst, self.channel), what)
                nomVarFlat = 0
                nomVarUp   = 0
                nomVarDn   = 0
                for w in what:
                    tfFlat = ROOT.TFile(self.systs["yields"]["file"].replace("[ROOTFILE]", self.options.rootfile).replace("[CHANNEL]", w if w != "bin1" else ""),"READ")
                    if w == 'SR':
                      nomFlatH = tfFlat.Get(self.systs["yields"]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", 'onecluster'))
                      upflatH = tfFlat.Get((self.systs[syst]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", 'onecluster') if not("total_background" in ss) else ("total_background" + "_" + syst)) + self.systs[syst]["up"])
                      if self.systs[syst]["down"] != "":
                          dnflatH = tfFlat.Get((self.systs[syst]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", 'onecluster') if not("total_background" in ss) else ("total_background" + "_" + syst)) + self.systs[syst]["down"])
                      else:
                          dnflatH = tfFlat.Get(self.systs["yields"]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", 'onecluster'))              
                    else:
                      nomFlatH = tfFlat.Get(self.systs["yields"]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", w))
                      upflatH = tfFlat.Get((self.systs[syst]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", w) if not("total_background" in ss) else ("total_background" + "_" + syst)) + self.systs[syst]["up"])
                      if self.systs[syst]["down"] != "":
                          dnflatH = tfFlat.Get((self.systs[syst]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", w) if not("total_background" in ss) else ("total_background" + "_" + syst)) + self.systs[syst]["down"])
                      else:
                          dnflatH = tfFlat.Get(self.systs["yields"]["match"].replace("$PROCESS", self.samples[ss]["name"]).replace("[VAR]", self.options.var).replace("$SYSTEMATIC",syst).replace("[CHANNEL]", w))
                    nomFlatH.SetDirectory(0)
                    upflatH.SetDirectory(0)
                    dnflatH.SetDirectory(0)
                    tfFlat.Close()
                    self.tf.cd()
                    for ibinFlat in range(1, nomFlatH.GetNbinsX()+1):
                        if not(ibinFlat in what[w]) or (ibinFlat >= (self.nbins+1) and options.integrateBins > 0): continue
                        nomVarFlat += nomFlatH.GetBinContent(ibinFlat)
                        nomVarUp += upflatH.GetBinContent(ibinFlat)
                        nomVarDn += dnflatH.GetBinContent(ibinFlat)
                varFlatUp = nomVarUp/max(nomVarFlat, 1e-5)
                varFlatDn = nomVarDn/max(nomVarFlat, 1e-5)
                #print("For systematic %s on channel %s flattened uncertainty is +%1.3f/-%1.3f"%(syst, self.channel, varFlatUp, varFlatDn))
                for ibin in range(1, self.nbins + 1):
                  self.th1s_perbin[ibin][ss + syst + "Up"] = self.th1s_perbin[ibin][ss].Clone("%s_%sbin%i_%sUp"%(ss, self.channel, ibin, name))
                  self.th1s_perbin[ibin][ss + syst + "Up"].Scale(varFlatUp)
                  self.th1s_perbin[ibin][ss + syst + "Down"] = self.th1s_perbin[ibin][ss].Clone("%s_%sbin%i_%sDown"%(ss, self.channel, ibin, name))
                  self.th1s_perbin[ibin][ss + syst + "Down"].Scale(varFlatDn)
                  if self.th1s_perbin[ibin][ss + syst + "Up"].GetBinContent(1) < 1e-4: self.th1s_perbin[ibin][ss + syst + "Up"].SetBinContent(1, 1e-4)
                  if self.th1s_perbin[ibin][ss + syst + "Down"].GetBinContent(1) < 1e-4: self.th1s_perbin[ibin][ss + syst + "Down"].SetBinContent(1, 1e-4)
                self.th1s_perbin[ibin][ss + syst + "Up"].SetDirectory(0)
                self.th1s_perbin[ibin][ss + syst + "Down"].SetDirectory(0)

            else:
              for ibin in range(1, self.nbins + 1):
                if (options.integrateBins == -1) or (ibin < options.integrateBins):
                  self.th1s_perbin[ibin][ss + syst + "Up"] = ROOT.TH1F("%s_%sbin%i_%sUp"%(ss, self.channel, ibin, name), "%s_%sbin%i_%sUp"%(ss, self.channel, ibin, name), 1, 0 , 1)
                  self.th1s_perbin[ibin][ss + syst + "Up"].SetBinContent(1, max(0.0001, factor*newup.GetBinContent(ibin)))
                  self.th1s_perbin[ibin][ss + syst + "Up"].SetBinError(1, newup.GetBinError(ibin)*factor)
                  self.th1s_perbin[ibin][ss + syst + "Down"] = ROOT.TH1F("%s_%sbin%i_%sDown"%(ss, self.channel, ibin, name), "%s_%sbin%i_%sDown"%(ss, self.channel, ibin, name), 1, 0 , 1)
                  self.th1s_perbin[ibin][ss + syst + "Down"].SetBinContent(1, max(0.0001, factor*newdn.GetBinContent(ibin) if self.systs[syst]["down"] != "" else self.th1s_perbin[ibin][ss].GetBinContent(1)))
                  self.th1s_perbin[ibin][ss + syst + "Down"].SetBinError(1, factor*newdn.GetBinError(ibin) if self.systs[syst]["down"] != "" else self.th1s_perbin[ibin][ss].GetBinError(1))
                elif (options.integrateBins > 0) or (ibin == options.integrateBins):
                  self.th1s_perbin[ibin][ss + syst + "Up"] = ROOT.TH1F("%s_%sbin%i_%sUp"%(ss, self.channel, ibin, name), "%s_%sbin%i_%sUp"%(ss, self.channel, ibin, name), 1, 0 , 1)
                  self.th1s_perbin[ibin][ss + syst + "Down"] = ROOT.TH1F("%s_%sbin%i_%sDown"%(ss, self.channel, ibin, name), "%s_%sbin%i_%sDown"%(ss, self.channel, ibin, name), 1, 0 , 1)
                  for iOldBin in range(options.integrateBins, newdn.GetNbinsX()+1):
                      self.th1s_perbin[ibin][ss + syst + "Up"].SetBinContent(1, self.th1s_perbin[ibin][ss + syst + "Up"].GetBinContent(1) + max(0.0001, factor*newup.GetBinContent(iOldBin)))
                      self.th1s_perbin[ibin][ss + syst + "Up"].SetBinError(1, (newup.GetBinError(iOldBin)*factor**2 + self.th1s_perbin[ibin][ss + syst + "Up"].GetBinError(1)**2)**0.5)
                      self.th1s_perbin[ibin][ss + syst + "Down"].SetBinContent(1, self.th1s_perbin[ibin][ss + syst + "Down"].GetBinContent(1) + max(0.0001, factor*newdn.GetBinContent(iOldBin)))
                      self.th1s_perbin[ibin][ss + syst + "Down"].SetBinError(1, (newdn.GetBinError(iOldBin)*factor**2 + self.th1s_perbin[ibin][ss + syst + "Down"].GetBinError(1)**2)**0.5)
 
      if options.ManualMC:
        for ibin in range(1, self.nbins + 1):
          willWrite = False
          toWrite = ""
          signal = self.th1s[s]
          if signal.GetBinError(ibin) == 0:
            rawMC = 0
          else:
            rawMC = (signal.GetBinContent(ibin)/signal.GetBinError(ibin))**2
          toWrite = "ManualMCStats_bin"+ str(ibin) + "_" + str(options.year) + "_" + str(options.region) + "_" + self.channel + " gmN " + str(int(round(rawMC))) + " "
          for j in range(ibin-1):
            toWrite += '- - '
          if rawMC == 0 and self.channel in ["B1","B2","SR"]:
            # Now we run the choppy binning search algorithm
            needsgmN = False
            if (ibin == 1) and (2*signal.GetBinContent(2)-signal.GetBinContent(3)) > 0 and (signal.GetBinContent(2) > 0 and signal.GetBinContent(3)> 0) : needsgmN = True
            elif (ibin == self.nbins) and  (2*signal.GetBinContent(self.nbins-1)-signal.GetBinContent(self.nbins-2)) > 0 and  (signal.GetBinContent(self.nbins-1) > 0 and signal.GetBinContent(self.nbins-2)> 0): needsgmN = True
            elif (ibin != 1) and (ibin != self.nbins) and (signal.GetBinContent(ibin+1) > 0) and (signal.GetBinContent(ibin-1) > 0): needsgmN = True
            if needsgmN:
              willWrite = True
              print("Choppiness detected in bin %i for channel %s, added gmN 0"%(ibin, self.channel))
              if options.year == '2018':
                val = 5252.99 / 150000
              elif options.year == '2017':
                val = 3639.48 / 150000
              elif options.year == '2016':
                val = 1438.31 / 75000
              elif options.year == '2016APV':
                val = 1745.15 / 75000
              toWrite += str(val) + ' - '
          elif rawMC != 0:
            willWrite = True
            toWrite += str(signal.GetBinContent(ibin)/rawMC) + ' - '
          else:
            toWrite += '- - '
          for j in range(self.nbins - ibin):
            toWrite += '- - '
          toWrite += '\n'
          if willWrite:
              newcard.write(toWrite)
      else:
        newcard.write("* autoMCStats 0 1\n")
      rout = ROOT.TFile(self.output + "/" + options.region + "%s_"%self.channel + self.samples[s]["name"] + "_shapes.root", "RECREATE")
      rout.cd()
      for ibin in range(1, self.nbins+1):
        for h in self.th1s_perbin[ibin]:
          if s in h or "background" in h or "data" in h:
            self.th1s_perbin[ibin][h].Write()
      for ibin in range(1, self.nbins+1):
        if not("SR" in self.channel) and not(options.MCABCD):
          newcard.write("r%s%s_%s rateParam %s %s %1.3f [%1.3f,%1.3f]\n"%(self.options.year,self.options.region, bins[(ibin-1)*2], bins[(ibin-1)*2], "total_background", max(self.yields[ibin]["data_obs"],1), 0.001, max(10.,3*self.yields[ibin]["data_obs"])))
      rout.Close()


  def collectYields(self):
    if self.systs["yields"]["type"] == "yields":
      self.collectYieldsCutAndCount()
      self.nbins = 1
    elif self.systs["yields"]["type"] == "yieldsWithShapes" and not(options.ABCD):
      self.collectYieldsShapes()
      #self.collectYieldsPerBin()
      self.nbins = 1
    elif options.ABCD:
      self.collectYieldsPerBin()
    
  def collectYieldsCutAndCount(self): #TODO
    self.isCC = True
    pass
  def collectYieldsShapes(self):
    print("Collecting shape histograms...")
    self.isShape = True
    self.tf = ROOT.TFile(self.systs["yields"]["file"].replace("[ROOTFILE]", self.options.rootfile).replace("[CHANNEL]", self.channel if self.channel != "bin1" else ""),"READ")
    self.th1s   = {}
    self.yields = {}
    toDelete = []
    for s in self.samples:
      #print("....%s"%s, self.systs["yields"]["match"].replace("$PROCESS", self.samples[s]["name"]).replace("[VAR]", options.var).replace("[CHANNEL]",self.channel) if not ("total_background" == s) else "total_background")
      newth1 = self.tf.Get(self.systs["yields"]["match"].replace("$PROCESS", self.samples[s]["name"]).replace("[VAR]", options.var).replace("[CHANNEL]",self.channel) if not ("total_background" == s) else "total_background") 
      if s != "data" or "data_obs" in self.th1s:
        self.th1s[s] = copy.deepcopy(self.th1s["data_obs"] if s=="data" else newth1.Clone(s))
      else:
        self.th1s["data_obs"] = copy.deepcopy(newth1.Clone("data_obs"))

      self.yields[s] = newth1.Integral()
      if (self.yields[s] == 0 and "SUEP" in s):
        self.yields[s] = 0.0001*self.th1s[s].GetNbinsX()
        for ibin in range(1, self.th1s[s].GetNbinsX()+1):
          self.th1s[s].SetBinContent(ibin, 0.0001)
      if (self.yields[s] <= 0 and not("SUEP" in s)) or s == "data":
        print("Process %s has 0 yields or is data, skipping..."%s)
        if not s=="data": del self.th1s[s]
        if not s=="data": del self.yields[s]
        toDelete.append(s)
        #del self.samples[s]
        continue
      if not( "isSig" in self.samples[s]):
        self.backgr.append(s)
      else:
        if self.samples[s]["isSig"]:
          self.signals.append(s)
          continue
        else:
          self.backgr.append(s)
      if self.options.blind or self.options.blind_withS:  
        if not("data_obs" in self.th1s):
          self.th1s["data_obs"] = copy.deepcopy(self.th1s[s].Clone("data_obs"))
        else:
          self.th1s["data_obs"].Add(self.th1s[s])
    for d in toDelete:
      del self.samples[d]
    #print(self.signals, self.backgr)
  def collectYieldsPerBin(self):
    print("Collecting shape histograms for channel %s..."%self.channel)
    self.isShape = True
    print("...file is %s"%(self.systs["yields"]["file"].replace("[ROOTFILE]", self.options.rootfile).replace("[CHANNEL]", self.channel if self.channel != "bin1" else "")))
    #self.tf = ROOT.TFile(self.systs["yields"]["file"].replace("[ROOTFILE]", self.options.rootfile).replace("[CHANNEL]", self.channel if self.channel != "bin1" else ""),"READ")
    self.tf = ROOT.TFile(self.systs["yields"]["file"].replace("[ROOTFILE]", self.options.rootfile).replace("[CHANNEL]", self.channel if self.channel != "bin1" else ""),"READ")
    self.th1s        = {}
    self.th1s_perbin = {}
    self.yields      = {}
    self.nbins = -1
    # First the nominal yields
    for s in self.samples:
      if not(self.samples[s]["isSig"]) and not(s == "data") and not(options.MCABCD): continue
      if self.samples[s]["name"] == "data" and (self.options.blind or self.options.blind_withS): continue
      if self.options.thispoint:
        if s != self.options.thispoint and not(self.samples[s]["name"] == "data" ): continue
      if self.channel == 'SR':
        #print(self.systs["yields"]["match"].replace("$PROCESS", self.samples[s]["name"]).replace("[VAR]", self.options.var).replace("[CHANNEL]", 'SR') if not("total_background" in s) else "total_background")
        newth1 = self.tf.Get(self.systs["yields"]["match"].replace("$PROCESS", self.samples[s]["name"]).replace("[VAR]", self.options.var).replace("[CHANNEL]", 'onecluster') if not("total_background" in s) else "total_background")
      else:
        #print(self.systs["yields"]["match"].replace("$PROCESS", self.samples[s]["name"]).replace("[VAR]", self.options.var).replace("[CHANNEL]", self.channel) if not("total_background" in s) else "total_background")
        newth1 = self.tf.Get(self.systs["yields"]["match"].replace("$PROCESS", self.samples[s]["name"]).replace("[VAR]", self.options.var).replace("[CHANNEL]", self.channel) if not("total_background" in s) else "total_background")
      if options.integrateBins != -1:
        if newth1.GetNbinsX() > options.integrateBins:
          print("Integrating bins")
          newN = options.integrateBins
          tempth1 = ROOT.TH1F(newth1.GetName() + "_int", newth1.GetName() + "_int", newN, 0, newN)
          for ibin in range(1,newth1.GetNbinsX()+1):
            if ibin < options.integrateBins:
              tempth1.SetBinContent(ibin, newth1.GetBinContent(ibin))
              tempth1.SetBinError(ibin, newth1.GetBinError(ibin))
            else:
              tempth1.SetBinContent(newN, newth1.GetBinContent(ibin)+ tempth1.GetBinContent(newN))
              tempth1.SetBinError(newN, (newth1.GetBinError(ibin)**2 + tempth1.GetBinError(newN)**2)**0.5)
          newth1 = tempth1
      #print(self.systs["yields"]["match"].replace("$PROCESS", self.samples[s]["name"]).replace("[VAR]", self.options.var).replace("[CHANNEL]", self.channel) if not("total_background" in s) else "total_background")
      self.th1s[s]   = copy.deepcopy(newth1.Clone("data_obs") if s=="data" else newth1.Clone(s))
      self.nbins = self.th1s[s].GetNbinsX()
      if not( "isSig" in self.samples[s]):
        self.backgr.append(s)
      else:
        if self.samples[s]["isSig"]:
          self.signals.append(s)
          continue
        else:
          self.backgr.append(s)
    if self.options.ABCD and not options.MCABCD:
      self.th1s["total_background"] = copy.deepcopy(self.tf.Get("total_background").Clone("total_background"))
      self.backgr.append("total_background")
    if self.options.blind: 
      self.th1s["data_obs"] = copy.deepcopy(self.tf.Get("total_background").Clone("data_obs"))
      self.data.append("data_obs")
    if self.options.blind_withS:
      total_background = self.tf.Get("total_background")
      if self.channel == 'SR':
        signal = self.tf.Get("leadclustertracks_onecluster_" + options.thispoint)
      else:
        signal = self.tf.Get("leadclustertracks_" + self.channel + "_"+options.thispoint)
      data_obs = total_background.Clone("data_obs")
      data_obs.Add(signal)
      self.th1s["data_obs"] = copy.deepcopy(data_obs)
      self.data.append("data_obs")
    elif "data" in self.th1s.keys():
      self.th1s["data_obs"] = self.th1s["data"]
      self.data.append("data_obs")
    for ibin in range(1,self.nbins+1):
      self.yields[ibin] = {} 
      self.th1s_perbin[ibin] = {}
      for s in self.signals + self.backgr + self.data:
        if options.MCABCD:
          factor = 1
          if s in self.signals: factor = options.scaleS
          self.yields[ibin][s]      = self.th1s[s].GetBinContent(ibin)
          self.th1s_perbin[ibin][s] = ROOT.TH1F("%s_%sbin%i"%(s, self.channel, ibin), "%s_%sbin%i"%(s, self.channel, ibin), 1, 0 , 1)
          self.th1s_perbin[ibin][s].SetBinContent(1, max(0.0001,self.yields[ibin][s]*(factor if self.yields[ibin][s] > 0.00011 else 1)))
          self.th1s_perbin[ibin][s].SetBinError(1, min(self.yields[ibin][s], max(0.,self.th1s[s].GetBinError(ibin) )))
          #if s in self.backgr:
          #  self.th1s_perbin[ibin][s].SetBinError(1, 0.)

        else:
          factor = 1
          if s in self.signals: factor = options.scaleS
          self.yields[ibin][s]      = max(0.1 if s in self.backgr else 0.0001, self.th1s[s].GetBinContent(ibin))
          self.th1s_perbin[ibin][s] = ROOT.TH1F("%s_%sbin%i"%(s, self.channel, ibin), "%s_%sbin%i"%(s, self.channel, ibin), 1, 0 , 1)
          self.th1s_perbin[ibin][s].SetBinContent(1, self.yields[ibin][s]*(factor if self.yields[ibin][s] > 0.00011 else 1) if not s in self.backgr else 1.)
          self.th1s_perbin[ibin][s].SetBinError(1, min(self.yields[ibin][s], max(0.,self.th1s[s].GetBinError(1) )))
          if s in self.backgr:
            self.th1s_perbin[ibin][s].SetBinError(1, 0.)

def combineDatacards(samples, options, output, nbins):
  # Get all ABCD cards, combine them and add the ABCD stuff
  os.chdir(output)
  for s in samples:
    if not(samples[s]["isSig"]):  continue
    if options.thispoint:
      if s != options.thispoint: continue
    names = ["%s=%s%s"%(channel,options.region,channel) + "_" + samples[s]["name"] + ".txt" for channel in ["SR","B1","B2","C1","C2","D1","D2","E1","E2"]]
    os.system("rm %s"%("Combined" + options.region + "_"+ samples[s]["name"] + ".txt"))
    os.system("combineCards.py %s >> %s"%(" ".join(names), "Combined" + options.region + "_"+ samples[s]["name"] + ".txt"))
    newcard = open("Combined" + options.region +"_" + samples[s]["name"] + ".txt",'a')
    if options.MCABCD: continue
    for ibin in range(1, nbins+1):
      newcard.write("r%s%s_%s rateParam %s %s %s %s\n"%(options.year,options.region, "SRbin%i"%ibin, "SR_SRbin%i"%ibin, "total_background", "(@0*@0*@2*@2*@5*@5*@6*@6)/(@1*@3*@7*@4*@4*@4*@4)", ",".join(["r%s%s_%s"%(options.year,options.region, "%sbin%i"%(chan,ibin if "B" in chan else 1)) for chan in ["B1","B2", "C1","C2", "D1","D2", "E1","E2"]])))
    if options.floatB:
      newcard.write("r%s%s rateParam SR* total_background 1"%(options.region, options.year[2:]))

if __name__ == "__main__":
  print("Starting plotting script...")
  from optparse import OptionParser
  parser = OptionParser(usage="%prog [options] samples.py systs.py output") 
  parser.add_option("--ms", dest="multisignal", action="store_true", default=False, help="Activate to have all signals in same card (default is different cards)")
  parser.add_option("--blind", dest="blind", action="store_true", default=False, help="Activate for blinding (no data)")
  parser.add_option("--blind_withS", dest="blind_withS", action="store_true", default=False, help="Activate for blinding (no data) with Signal on top of background")
  parser.add_option("--rootfile", dest="rootfile", default="", help="ROOT file with the input shapes")
  parser.add_option("--var", dest="var", default="", help="Variable name in the plots file used to produce the rootfile")
  parser.add_option("--ABCD", dest="ABCD", default=False, action="store_true", help="Use ABCD background instead of directly MC")
  parser.add_option("--floatB", dest="floatB", default=False, action="store_true", help="Also free float the background")
  parser.add_option("--year", dest="year", default="2018", help="Which year the card is from")
  parser.add_option("--region", dest="region", default="SR", help="Which region the card is from")
  parser.add_option("--thispoint", dest="thispoint", default=None, help="Run only this point")
  parser.add_option("--doAll", dest="doAll", default=False, action="store_true", help="Run also signals for which doPlot is set to False in the samples file")
  parser.add_option("--scaleS", dest="scaleS", default=1, type=float, help="Scale signal by this prefactor for alternative x-section (or real cross section if you messed up the plotting) checks")
  parser.add_option("--exclude", dest="exclude", default=[], action="append", help="Exclude samples matching this")
  parser.add_option("--MCABCD", dest="MCABCD", default=False, action="store_true", help="Save MC predictions for ABCD regions as well")
  parser.add_option("--ManualMCStats", dest="ManualMC", default=False, action="store_true", help="Don't use autoMCStats - Poissonian")
  parser.add_option("--integrateBins", dest="integrateBins", type=int, default=-1, help="From this bin onwards, integrate into a single bin")
  (options, args) = parser.parse_args()
  samplesFile   = imp.load_source("samples",args[0])
  systsFile     = imp.load_source("systematicsAndShapes", args[1])
  outputFolder  = args[2]
  if (options.blind or options.blind_withS) and "data" in samplesFile.samples:
    del samplesFile.samples["data"]
  samp = samplesFile.samples.keys()
  for s in samp:
    if "doPlot" in samplesFile.samples[s] and not(options.doAll):
      if not(samplesFile.samples[s]["doPlot"]):
        del samplesFile.samples[s]

  if options.exclude != []:
    newsamples = {}
    for ss in samplesFile.samples.keys():
      add = True
      for s in options.exclude:
        if re.match(s, ss):
          print("Will exclude sample:", ss)
          add = False
      if add:
        newsamples[ss] = samplesFile.samples[ss]
    samplesFile.samples = newsamples

  if not (options.ABCD): #Build MC based cards, simple and direct
    dM = datacardMaker(samplesFile.samples, systsFile.systematicsAndShapes, outputFolder, options)
    dM.createDatacards()
  else: # If ABCD, one card per channel
    nToSave = -1
    for ch in systsFile.systematicsAndShapes["yields"]["extendedABCD"]:
      dM =  datacardMaker(samplesFile.samples, systsFile.systematicsAndShapes, outputFolder, options, channel = ch)
      n = dM.createDatacards()
      del dM
      if ch == "SR": nToSave = n
    combineDatacards(samplesFile.samples, options, outputFolder, nToSave)
