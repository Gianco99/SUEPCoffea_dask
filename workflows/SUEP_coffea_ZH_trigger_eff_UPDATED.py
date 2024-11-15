"""
SUEP_coffea_ZH.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer, 2021
"""

import os
import pathlib
import shutil
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import fastjet
from coffea import hist, processor
import vector
from typing import List, Optional
vector.register_awkward()

class SUEP_cluster(processor.ProcessorABC):
    def __init__(self, isMC: int, era: int, sample: str,  do_syst: bool, syst_var: str, weight_syst: bool, SRonly: bool, output_location: Optional[str], isDY: Optional[bool]) -> None:
        self.SRonly = SRonly
        self.output_location = output_location
        self.isDY = isDY # We need to save this to remove the overlap between the inclusive DY sample and the pT binned ones
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}
        #Set up for the histograms
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def rho(self, number, jet, tracks, deltaR, dr=0.05):
        r_start = number*dr
        r_end = (number+1)*dr
        ring = (deltaR > r_start) & (deltaR < r_end)
        rho_values = ak.sum(tracks[ring].pt, axis=1)/(dr*jet.pt)
        return rho_values

    def ak_to_pandas(self, jet_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(jet_collection):
            prefix = self.prefixes.get(field, "")
            if len(prefix) > 0:
                for subfield in ak.fields(jet_collection[field]):
                    output[f"{prefix}_{subfield}"] = ak.to_numpy(
                        jet_collection[field][subfield]
                    )
            else:
                if not(isinstance(ak.to_numpy(jet_collection[field])[0],np.ndarray)):
                    output[field] = ak.to_numpy(jet_collection[field])
                else:
                    temp =  ak.to_numpy(jet_collection[field])
                    output[field] = [[k for k in kk] for kk in temp]
        return output

    def h5store(self, store: pd.HDFStore, df: pd.DataFrame, fname: str, gname: str, **kwargs: float) -> None:
        store.put(gname, df)
        store.get_storer(gname).attrs.metadata = kwargs

    def save_dfs(self, dfs, df_names, fname=None):
        if not(fname): fname = "out.hdf5"
        subdirs = []
        store = pd.HDFStore(fname)
        if self.output_location is not None:
            # pandas to hdf5
            for out, gname in zip(dfs, df_names):
                if self.isMC:
                    metadata = dict(gensumweight=self.gensumweight,era=self.era, mc=self.isMC,sample=self.sample)
                    #metadata.update({"gensumweight":self.gensumweight})
                else:
                    metadata = dict(era=self.era, mc=self.isMC,sample=self.sample)

                store_fin = self.h5store(store, out, fname, gname, **metadata)

            store.close()
            self.dump_table(fname, self.output_location, subdirs)
        else:
            print("self.output_location is None")
            store.close()

    def dump_table(self, fname: str, location: str, subdirs: Optional[List[str]] = None) -> None:
        subdirs = subdirs or []
        xrd_prefix = "root://"
        pfx_len = len(xrd_prefix)
        xrootd = False
        if xrd_prefix in location:
            try:
                import XRootD
                import XRootD.client

                xrootd = True
            except ImportError:
                raise ImportError(
                    "Install XRootD python bindings with: conda install -c conda-forge xroot"
                )
        local_file = (
            os.path.abspath(os.path.join(".", fname))
            if xrootd
            else os.path.join(".", fname)
        )
        merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
        destination = (
            location + merged_subdirs + f"/{fname}"
            if xrootd
            else os.path.join(location, os.path.join(merged_subdirs, fname))
        )
        if xrootd:
            copyproc = XRootD.client.CopyProcess()
            copyproc.add_job(local_file, destination)
            copyproc.prepare()
            copyproc.run()
            client = XRootD.client.FileSystem(
                location[: location[pfx_len:].find("/") + pfx_len]
            )
            status = client.locate(
                destination[destination[pfx_len:].find("/") + pfx_len + 1 :],
                XRootD.client.flags.OpenFlags.READ,
            )
            assert status[0].ok
            del client
            del copyproc
        else:
            dirname = os.path.dirname(destination)
            if not os.path.exists(dirname):
                pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            if os.path.isfile(destination):
                if not os.path.samefile(local_file, destination):
                    shutil.copy2(local_file, destination)
                else:
                  fname = "condor_" + fname
                  destination = os.path.join(location, os.path.join(merged_subdirs, fname))
                  shutil.copy2(local_file, destination)
            else:
                shutil.copy2(local_file, destination)
            assert os.path.isfile(destination)
        pathlib.Path(local_file).unlink()


    def selectByFilters(self, events):
        ### Apply MET filter selection (see https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2)
        if self.era == 2018 or self.era == 2017 or self.era == 1999:
           cutAnyFilter = (events.Flag.goodVertices) | (events.Flag.globalSuperTightHalo2016Filter) | (events.Flag.HBHENoiseFilter) | (events.Flag.HBHENoiseIsoFilter) | (events.Flag.EcalDeadCellTriggerPrimitiveFilter) | (events.Flag.BadPFMuonFilter) | (events.Flag.BadPFMuonDzFilter) | (events.Flag.eeBadScFilter) | (events.Flag.ecalBadCalibFilter)
        if self.era == 2016:
           cutAnyFilter = (events.Flag.goodVertices) | (events.Flag.globalSuperTightHalo2016Filter) | (events.Flag.HBHENoiseFilter) | (events.Flag.HBHENoiseIsoFilter) | (events.Flag.EcalDeadCellTriggerPrimitiveFilter) | (events.Flag.BadPFMuonFilter) | (events.Flag.BadPFMuonDzFilter) | (events.Flag.eeBadScFilter)
        return events[cutAnyFilter]

    def selectByJets(self, events, leptons = [],  extraColls = []):
        # These are just standard jets, as available in the nanoAOD
        Jets = ak.zip({
            "pt": events.Jet.pt,
            "eta": events.Jet.eta,
            "phi": events.Jet.phi,
            "mass": events.Jet.mass,
            "btag": events.Jet.btagDeepFlavB,
            "jetId": events.Jet.jetId
        }, with_name="Momentum4D")
        # Minimimum pT, eta requirements + jet-lepton recleaning
        jetCut = (Jets.pt > 30) & (abs(Jets.eta)<2.5) & (Jets.deltaR(leptons[:,0])>= 0.4) & (Jets.deltaR(leptons[:,1])>= 0.4) & (Jets.jetId >= 6)
        jets = Jets[jetCut]
        # The following is the collection of events and of jets
        return events, jets, [coll for coll in extraColls]

    def selectByTrigger(self, events, extraColls = []):

        self.METElecTrig_a = [-1]*len(self.electrons)
        self.METElecTrig_b = [-1]*len(self.electrons)
        self.SingleElecTrig = [-1]*len(self.electrons)
        self.DoubleElecTrig = [-1]*len(self.electrons)

        self.METMuonTrig_a = [-1]*len(self.muons)
        self.METMuonTrig_b = [-1]*len(self.muons)
        self.SingleMuonTrig = [-1]*len(self.muons)
        self.DoubleMuonTrig = [-1]*len(self.muons)

        ### Apply trigger selection
        if self.era == 2018:
            self.METElecTrig_a = np.where(events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight, 1, 0)
            self.METElecTrig_b = np.where(events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight, 1, 0)
            self.SingleElecTrig = np.where(events.HLT.Ele32_WPTight_Gsf, 1, 0)
            self.DoubleElecTrig = np.where(events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL, 1, 0)
                        
            self.METMuonTrig_a = np.where(events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight, 1, 0)
            self.METMuonTrig_b = np.where(events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight, 1, 0)
            self.SingleMuonTrig = np.where(events.HLT.IsoMu24, 1, 0)
            self.DoubleMuonTrig = np.where(events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8, 1, 0)

            return events
        
        if self.era == 2017:
            self.METElecTrig_a = np.where(events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight, 1, 0)
            self.METElecTrig_b = np.where(events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight, 1, 0)
            self.SingleElecTrig = np.where(events.HLT.Ele35_WPTight_Gsf, 1, 0)
            self.DoubleElecTrig = np.where(events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL, 1, 0)
                        
            self.METMuonTrig_a = np.where(events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight, 1, 0)
            self.METMuonTrig_b = np.where(events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight, 1, 0)
            self.SingleMuonTrig = np.where(events.HLT.IsoMu27, 1, 0)
            self.DoubleMuonTrig = np.where(events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8, 1, 0)

            return events

        if self.era == 2016:
            self.METElecTrig_a = np.where(events.HLT.PFMETNoMu90_PFMHTNoMu90_IDTight, 1, 0)
            self.METElecTrig_b = np.where(events.HLT.PFHT300_PFMET110, 1, 0)
            self.SingleElecTrig = np.where(events.HLT.Ele27_WPTight_Gsf, 1, 0)
            self.DoubleElecTrig = np.where(events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ, 1, 0)
                        
            self.METMuonTrig_a = np.where(events.HLT.PFMETNoMu90_PFMHTNoMu90_IDTight, 1, 0)
            self.METMuonTrig_b = np.where(events.HLT.PFHT300_PFMET110, 1, 0)
            self.SingleMuonTrig = np.where(events.HLT.IsoMu24, 1, 0)
            self.DoubleMuonTrig = np.where(events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ, 1, 0)

            return events
        return events

    def selectByLeptons(self, events, extraColls = []):
    ###lepton selection criteria--4momenta collection for plotting

        muons = ak.zip({
            "pt": events.Muon.pt,
            "eta": events.Muon.eta,
            "phi": events.Muon.phi,
            "mass": events.Muon.mass,
            "charge": events.Muon.pdgId/(-13),
        }, with_name="Momentum4D")

        electrons = ak.zip({
            "pt": events.Electron.pt,
            "eta": events.Electron.eta,
            "phi": events.Electron.phi,
            "mass": events.Electron.mass,
            "charge": events.Electron.pdgId/(-11),
        }, with_name="Momentum4D")

        ###  Some very simple selections on ID ###
        ###  Muons: loose ID + dxy dz cuts mimicking the medium prompt ID https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2
        ###  Electrons: loose ID + dxy dz cuts for promptness https://twiki.cern.ch/twiki/bin/view/CMS/EgammaCutBasedIdentification
        cutMuons     = (events.Muon.looseId) & (events.Muon.pt >= 10) & (abs(events.Muon.dxy) <= 0.02) & (abs(events.Muon.dz) <= 0.1) & (events.Muon.pfIsoId >= 2)
        cutElectrons = (events.Electron.cutBased >= 2) & (events.Electron.pt >= 15) & (events.Electron.mvaFall17V2Iso_WP90) & ( abs(events.Electron.dxy) < 0.05 + 0.05*(events.Electron.eta > 1.479)) & (abs(events.Electron.dz) <  0.10 + 0.10*(events.Electron.eta > 1.479)) & ((abs(events.Electron.eta) < 1.444) | (abs(events.Electron.eta) > 1.566))

        # Removed to test trigger efficiencies -

        ### Apply the cuts
        # Object selection. selMuons contain only the events that are filtered by cutMuons criteria.

        selMuons     = muons[cutMuons]
        selElectrons = electrons[cutElectrons]

        ### Now global cuts to select events. Notice this means exactly two leptons with pT >= 10, and the leading one pT >= 25

        # cutHasTwoMuons imposes three conditions:
        #  First, number of muons (axis=1 means column. Each row is an event.) in an event is 2.
        #  Second, pt of the muons is greater than 25.
        #  Third, Sum of charge of muons should be 0. (because it originates from Z

        cutHasTwoMuons = (ak.num(selMuons, axis=1)==2) & (ak.sum(selMuons.charge,axis=1) == 0) & (ak.max(selMuons.pt, axis=1, mask_identity=False) >= 25)
        cutHasTwoElecs = (ak.num(selElectrons, axis=1)==2) & (ak.sum(selElectrons.charge,axis=1) == 0) & (ak.max(selElectrons.pt, axis=1, mask_identity=False) >= 25)
        cutTwoLeps     = ((ak.num(selElectrons, axis=1)+ak.num(selMuons, axis=1)) < 3)
        cutHasTwoLeps  = ((cutHasTwoMuons) | (cutHasTwoElecs)) & cutTwoLeps
        ### Cut the events, also return the selected leptons for operation down the line
        events = events[cutHasTwoLeps]
        selElectrons = selElectrons[cutHasTwoLeps]
        selMuons = selMuons[cutHasTwoLeps]

        return events, selElectrons, selMuons
    
    def selectByTracks(self, events, leptons, extraColls = []):
        ### PARTICLE FLOW CANDIDATES ###
        # Every particle in particle flow (clean PFCand matched to tracks collection)
        Cands = ak.zip({
            "pt": events.PFCands.trkPt,
            "eta": events.PFCands.trkEta,
            "phi": events.PFCands.trkPhi,
            "mass": events.PFCands.mass
        }, with_name="Momentum4D")

        cutPF = (events.PFCands.fromPV > 1) & \
            (events.PFCands.trkPt >= 1) & \
            (abs(events.PFCands.trkEta) <= 2.5) & \
            (abs(events.PFCands.dz) < 0.05) & \
            (abs(events.PFCands.d0) < 0.05) & \
            (events.PFCands.puppiWeight > 0.1)
            #(events.PFCands.dzErr < 0.05)
        Cleaned_cands = ak.packed(Cands[cutPF])

	### LOST TRACKS ###
        # Unidentified tracks, usually SUEP Particles
        LostTracks = ak.zip({
            "pt": events.lostTracks.pt,
            "eta": events.lostTracks.eta,
            "phi": events.lostTracks.phi,
            "mass": 0.0
        }, with_name="Momentum4D")

        cutLost = (events.lostTracks.fromPV > 1) & \
            (events.lostTracks.pt >= 1) & \
            (abs(events.lostTracks.eta) <= 2.5) & \
            (abs(events.lostTracks.dz) < 0.05) & \
            (abs(events.lostTracks.d0) < 0.05) & \
            (events.lostTracks.puppiWeight > 0.1)
            #(events.lostTracks.dzErr < 0.05)
        Lost_Tracks_cands = ak.packed(LostTracks[cutLost])

        # dimensions of tracks = events x tracks in event x 4 momenta
        totalTracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)

        # Sorting out the tracks that overlap with leptons
        totalTracks = totalTracks[(totalTracks.deltaR(leptons[:,0])>= 0.4) & (totalTracks.deltaR(leptons[:,1])>= 0.4)]
        nTracks = ak.num(totalTracks,axis=1)
        return events, totalTracks, nTracks, [coll for coll in extraColls]

    def clusterizeTracks(self, events, tracks):
        # anti-kt, dR=1.5 jets
        nSmallEvents = 1000
        smallEvents = len(events) < nSmallEvents
        if not smallEvents:
          jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)        
          cluster = fastjet.ClusterSequence(tracks, jetdef)
          ak15_jets   = ak.with_name(cluster.inclusive_jets(min_pt=0),"Momentum4D") # These are the ak15_jets
          ak15_consts = ak.with_name(cluster.constituents(min_pt=0),"Momentum4D")   # And these are the collections of constituents of the ak15_jets
          return events, ak15_jets, ak15_consts
        else: #With few events/file the thing crashes because of FastJet so we are going to create "fake" events
          ncopies     = round(nSmallEvents/(len(events)))
          oldtracks   = tracks
          for i in range(ncopies):
            tracks = ak.concatenate([tracks, oldtracks], axis=0) # I.e. duplicate our events until we are feeding 1000 events to the clusterizer
          jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)
          cluster = fastjet.ClusterSequence(tracks, jetdef)
          ak15_jets   = ak.with_name(cluster.inclusive_jets(min_pt=0),"Momentum4D") # These are the ak15_jets
          ak15_consts = ak.with_name(cluster.constituents(min_pt=0),"Momentum4D")   # And these are the collections of constituents of the ak15_jets
          # But now we have to delete the repeated set of events
          return events, ak15_jets[:len(oldtracks)], ak15_consts[:len(oldtracks)]
        
    def shouldContinueAfterCut(self, events, out):
        #if debug: print("Conversion to pandas...")
        if True: # No need to filter it out
            if len(events) == 0:
                outdfs  = []
                outcols = []
                for channel in out.keys():
                    outcols.append(channel)
                    if out[channel][0] == {}:
                        outdfs = pd.DataFrame(['empty'], columns=['empty'])
                    else:
                        if self.isMC:
                            out[channel][0]["genweight"] = out[channel][1].genWeight[:]

                    if not isinstance(out[channel][0], pd.DataFrame):
                        out[channel][0] = self.ak_to_pandas(out[channel][0])

                return False
            else: return True


    def process(self, events):
        #print(events.event[0], events.luminosityBlock[0], events.run[0])
        # 255955082 94729 1
        #if not(events.event[0]==255955082 and events.luminosityBlock[0]==94729 and events.run[0]==1): return self.accumulator.identity()
        debug    = True  # If we want some prints in the middle
        chunkTag = "out_%i_%i_%i.hdf5"%(events.event[0], events.luminosityBlock[0], events.run[0]) #Unique tag to get different outputs per tag
        fullFile = self.output_location + "/" + chunkTag
        csvTag = "out_%i_%i_%i"%(events.event[0], events.luminosityBlock[0], events.run[0])

        print("Check file %s"%fullFile)
        if os.path.isfile(fullFile):
            print("SKIP")
            return self.accumulator.identity()

        # Main processor code


        # ------------------------------------------------------------------------------------
        # ------------------------------- DEFINE OUTPUTS -------------------------------------
        # ------------------------------------------------------------------------------------

        accumulator    = self.accumulator.identity()
        # Each track is one selection level
        outputs = {
            "twoleptons"  :[{},[]],   # Has Two Leptons, pT and Trigger requirements
        }

        # Data dependant stuff
        dataset = events.metadata['dataset']

        # ------------------------------------------------------------------------------------
        # ------------------------------- OBJECT LOADING -------------------------------------
        # ------------------------------------------------------------------------------------
        # MET filters
        if debug: print("Applying MET requirements.... %i events in"%len(events))
        self.events = self.selectByFilters(events)
        if not(self.shouldContinueAfterCut(self.events, outputs)): return accumulator # If we have no events, we simply stop
        if debug: print("%i events pass METFilter cuts. Applying lepton requirements...."%len(self.events))

        # Lepton selection
        self.events, self.electrons, self.muons = self.selectByLeptons(self.events)[:3]
        self.leptons = ak.concatenate([self.electrons, self.muons], axis=1)
        highpt_leptons = ak.argsort(self.leptons.pt, axis=1, ascending=False, stable=True)
        self.leptons = self.leptons[highpt_leptons]

        if not(self.shouldContinueAfterCut(self.events, outputs)): return accumulator # If we have no events, we simply stop

        # Trigger selection
        if debug: print("%i events pass lepton cuts. Applying trigger requirements...."%len(self.events))
        self.events = self.selectByTrigger(self.events,[self.electrons, self.muons])

        self.events, self.tracks = self.selectByTracks(self.events, self.leptons)[:2] # Again, we need leptons to clean the tracks

        self.events, self.clusters, self.constituents  = self.clusterizeTracks(self.events, self.tracks)[:3]

        highpt_clusters = ak.argsort(self.clusters.pt, axis=1, ascending=False, stable=True)
        self.clusters   = self.clusters[highpt_clusters]
        self.constituents = self.constituents[highpt_clusters]

        cutOneTrack = (ak.num(self.tracks) != 0)
        self.applyCutToAllCollections(cutOneTrack)

        cutOneCluster = (ak.num(self.clusters) != 0)
        self.applyCutToAllCollections(cutOneCluster)

        cutclusterpt60 = (self.clusters.pt[:,0] >= 60)
        self.applyCutToAllCollections(cutclusterpt60)

        # Remove empty events from electrons, muons, and trigger arrays
        selected_electrons_events = np.where(ak.num(self.electrons) != 0)
        selected_muon_events = np.where(ak.num(self.muons) == 2)

        self.electrons = self.electrons[selected_electrons_events]
        self.muons = self.muons[selected_muon_events]
        
        self.METElecTrig_a = self.METElecTrig_a[selected_electrons_events]
        self.METElecTrig_b = self.METElecTrig_b[selected_electrons_events]
        self.SingleElecTrig = self.SingleElecTrig[selected_electrons_events]
        self.DoubleElecTrig = self.DoubleElecTrig[selected_electrons_events]

        self.METMuonTrig_a = self.METMuonTrig_a[selected_muon_events]
        self.METMuonTrig_b = self.METMuonTrig_b[selected_muon_events]
        self.SingleMuonTrig = self.SingleMuonTrig[selected_muon_events]
        self.DoubleMuonTrig = self.DoubleMuonTrig[selected_muon_events]

        # Sort Electrons and Muons by pT
        highpt_electrons= ak.argsort(self.electrons.pt, axis=1, ascending=False, stable=True)
        self.electrons = self.electrons[highpt_electrons]

        highpt_muons = ak.argsort(self.muons.pt, axis=1, ascending=False, stable=True)
        self.muons = self.muons[highpt_muons]

        if not(self.shouldContinueAfterCut(self.events, outputs)): return accumulator
        outputs["twoleptons"] = [self.doAllPlots("twoleptons", csvTag, debug), self.events]
        if not(self.shouldContinueAfterCut(self.events, outputs)): return accumulator
        if debug: print("%i events pass twoleptons cuts. Doing more stuff..."%len(self.events))

        # ------------------------------------------------------------------------------
        # -------------------------------- SAVING --------------------------------------
        # ------------------------------------------------------------------------------
        todel = []
        self.SRonly = False
        if self.SRonly: # Lightweight, save only SR stuff
            for out in outputs:
                if not("SR"==out):
                    todel.append(out)
            for t in todel:
                del outputs[t]

        for out in outputs:
            if out in todel: continue
            if debug: print("Conversion to pandas...")
            if not isinstance(outputs[out][0], pd.DataFrame):
                if debug: print("......%s"%out)
                outputs[out][0] = self.ak_to_pandas(outputs[out][0])

        if debug: print("DFS saving....")

        self.save_dfs([outputs[key][0] for key in outputs], [key for key in outputs], chunkTag)

        return accumulator


    def applyCutToAllCollections(self, cut): # Cut has to by a selection applicable across all collections, i.e. something defined per event
        self.events    = self.events[cut]
        self.electrons = self.electrons[cut]
        self.muons     = self.muons[cut]
        self.leptons   = self.leptons[cut]
        self.tracks = self.tracks[cut]
        self.clusters = self.clusters[cut]
        self.METElecTrig_a = self.METElecTrig_a[cut]
        self.METElecTrig_b = self.METElecTrig_b[cut]
        self.SingleElecTrig = self.SingleElecTrig[cut]
        self.DoubleElecTrig = self.DoubleElecTrig[cut]

        self.METMuonTrig_a = self.METMuonTrig_a[cut]
        self.METMuonTrig_b = self.METMuonTrig_b[cut]
        self.SingleMuonTrig = self.SingleMuonTrig[cut]
        self.DoubleMuonTrig = self.DoubleMuonTrig[cut]



    def doAllPlots(self, channel, csvTag, debug=True):
        # ------------------------------------------------------------------------------
        # ------------------------------- PLOTTING -------------------------------------
        # ------------------------------------------------------------------------------
        out = {}
        # Define outputs for plotting
        if debug: print("Saving reco variables for channel %s"%channel)

        # Object: leptons
        df_electrons = pd.DataFrame({"LeadingPt" : self.electrons.pt[:,0], "SubleadingPt" : self.electrons.pt[:,1], "LeadingEta" : self.electrons.eta[:,0], "METTrig_a" : self.METElecTrig_a[:], "METTrig_b" : self.METElecTrig_b[:], "SingleElecTrig" : self.SingleElecTrig[:], "DoubleElecTrig" : self.DoubleElecTrig[:]})
        df_electrons.to_csv(self.output_location+"/"+csvTag+'_electrons.csv', index=False)

        df_muons = pd.DataFrame({"LeadingPt" : self.muons.pt[:,0], "SubleadingPt" : self.muons.pt[:,1], "LeadingEta" : self.muons.eta[:,0], "METTrig_a" : self.METMuonTrig_a[:], "METTrig_b" : self.METMuonTrig_b[:], "SingleMuonTrig" : self.SingleMuonTrig[:], "DoubleMuonTrig" : self.DoubleMuonTrig[:]})
        df_muons.to_csv(self.output_location+"/"+csvTag+'_muons.csv', index=False)

        '''
        out["leadlep_pt"] = self.leptons.pt[:,0]
        out["subleadlep_pt"] = self.leptons.pt[:,1]
        out["leadlep_eta"] = self.leptons.eta[:,0]
        out["subleadlep_eta"] = self.leptons.eta[:,1]
        out["leadlep_phi"] = self.leptons.phi[:,0]
        out["subleadlep_phi"] = self.leptons.phi[:,1]
        '''

        return out

    def postprocess(self, accumulator):
        return accumulator
