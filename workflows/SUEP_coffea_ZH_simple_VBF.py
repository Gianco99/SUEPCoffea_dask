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
import numpy as np
import fastjet
from coffea import processor, lookup_tools
import pickle5 as pickle
import vector
from typing import List, Optional
import correctionlib
from workflows.CMS_corrections.jetmet_utils import apply_jecs
import copy
import json

vector.register_awkward()

class SUEP_cluster(processor.ProcessorABC):
    def __init__(self, fileName: str, isMC: int, era: int, sample: str,  do_syst: bool, syst_var: str, weight_syst: bool, SRonly: bool, output_location: Optional[str], doOF: Optional[bool], isDY: Optional[bool]) -> None:
        self.fileName = fileName.strip()
        print(self.fileName)
        self.SRonly = SRonly
        self.output_location = output_location
        self.doOF = doOF
        self.isDY = isDY
        self.do_syst = (do_syst and isMC!=0)
        self.gensumweight = 1.0
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.isSignal = True
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def sphericity(self, events, particles, r):
        # In principle here we already have ak.num(particles) != 0
        # Some sanity replacements just in case the boosting broke
        px = ak.nan_to_num(particles.px, 0)
        py = ak.nan_to_num(particles.py, 0)
        pz = ak.nan_to_num(particles.pz, 0)
        p  = ak.nan_to_num(particles.p,  0)

        norm = np.squeeze(ak.sum(p ** r, axis=1, keepdims=True))
        s = np.array([[
                       ak.sum(px*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(px*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(px*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
                      ],
                      [
                       ak.sum(py*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(py*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(py*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
                      ],
                      [
                       ak.sum(pz*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(pz*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(pz*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
                       ]])
        s = np.squeeze(np.moveaxis(s, 2, 0),axis=3)
        s = np.nan_to_num(s, copy=False, nan=1., posinf=1., neginf=1.) 

        evals = np.sort(np.linalg.eigvals(s))
        # eval1 < eval2 < eval3
        return evals

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

    def selectByTrigger(self, events):
        ### Apply trigger selection
        if self.era == 2024:
           cutAnyHLT = (events.HLT.VBF_DiPFJet125_45_Mjj1050)
           return events[cutAnyHLT]
        return events
    
    def selectByJets(self, events):
        altJets = events.Jet
        if self.isMC:
            Jets = ak.zip({
                "pt": altJets.pt,
                "eta": altJets.eta,
                "phi": altJets.phi,
                "mass": altJets.mass,
                "btag": altJets.btagDeepFlavB,
                "hadronFlavour": altJets.hadronFlavour,
                "chEmEF": altJets.chEmEF,
                "neMultiplicity": altJets.neMultiplicity,
                "chMultiplicity": altJets.chMultiplicity,
                "chHEF": altJets.chHEF,
                "neEmEF": altJets.neEmEF,
                "muEF": altJets.muEF,
                "neHEF": altJets.neHEF,
                "btag": altJets.btagDeepFlavB,
                "nConstituents": altJets.nConstituents
            }, with_name="Momentum4D")
        else:
            Jets = ak.zip({
                "pt": altJets.pt,
                "eta": altJets.eta,
                "phi": altJets.phi,
                "mass": altJets.mass,
                "btag": altJets.btagDeepFlavB,
                "chEmEF": altJets.chEmEF,
                "neMultiplicity": altJets.neMultiplicity,
                "chMultiplicity": altJets.chMultiplicity,
                "chHEF": altJets.chHEF,
                "neEmEF": altJets.neEmEF,
                "muEF": altJets.muEF,
                "neHEF": altJets.neHEF,
                "btag": altJets.btagDeepFlavB,
                "nConstituents": altJets.nConstituents
            }, with_name="Momentum4D")

        # Apply initial pt and eta cuts with Jet ID from https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV#Recommendations_for_the_13_6_AN1
        jetCut = (Jets.pt > 45) & (abs(Jets.eta) < 5.0) & ( \
            ((abs(Jets.eta) < 2.6) & (Jets.chEmEF < 0.8) & (Jets.chMultiplicity > 0) & (Jets.chHEF > 0.01) & (Jets.nConstituents > 1) & (Jets.neEmEF < 0.9) & (Jets.muEF < 0.8) & (Jets.neHEF < 0.99)) |
             ((abs(Jets.eta) > 2.6) & (abs(Jets.eta) < 2.7) & (Jets.chEmEF < 0.8) & (Jets.neEmEF < 0.99) & (Jets.muEF < 0.8) & (Jets.neHEF < 0.9)) | \
             ((abs(Jets.eta) > 2.7) & (abs(Jets.eta) < 3.0) & (Jets.neHEF < 0.99)) | \
             ((abs(Jets.eta) > 3.0) & (Jets.neEmEF < 0.4) & (Jets.nConstituents - Jets.chMultiplicity >= 2)) \
        ) 
        jets = Jets[jetCut]

        # Require at least two AK4
        eventHasTwoJets = ak.num(jets) >= 2
        jets = jets[eventHasTwoJets]
        selectedEvents = events[eventHasTwoJets]

        # Create all possible unique jet pairs
        jetPairs = ak.combinations(jets, 2, fields=["jet1", "jet2"])

        # Check for valid pairs passing invariant mass, pt, and opposite hemispheres
        invariantMass = (jetPairs.jet1 + jetPairs.jet2).mass
        
        massCut = invariantMass > 1050
        ptCut = ((jetPairs.jet1.pt >= 125) & (jetPairs.jet2.pt >= 45)) | \
                ((jetPairs.jet2.pt >= 125) & (jetPairs.jet1.pt >= 45))
        oppositeEtaSigns = (jetPairs.jet1.eta * jetPairs.jet2.eta) < 0

        validPairs = massCut & ptCut & oppositeEtaSigns
        eventSelection = ak.any(validPairs, axis=1)

        # Filter events based on valid pairs
        selectedEvents = selectedEvents[eventSelection]
        jets = jets[eventSelection]
        jetPairs = jetPairs[validPairs]
        validJetPairs = jetPairs[eventSelection]
     
        # Select the highest mass pair
        mjj = (validJetPairs.jet1 + validJetPairs.jet2).mass
        sortingIndices = ak.argsort(-mjj, axis=1)
        sorted_validJetPairs = validJetPairs[sortingIndices]
        highestEnergyPairs = sorted_validJetPairs[:, 0]

        return selectedEvents, highestEnergyPairs, jets

    def selectByTracks(self, events, VBFCands):
        Cands = ak.zip({
            "pt": events.PFCands.trkPt,
            "eta": events.PFCands.trkEta,
            "phi": events.PFCands.trkPhi,
            "mass": events.PFCands.mass,
            "pdgId": events.PFCands.pdgId,
            "charge": events.PFCands.charge,
            "inclusivePt": events.PFCands.pt
        }, with_name="Momentum4D")

        cutPF = (events.PFCands.fromPV > 1) & \
            (events.PFCands.trkPt >= 1) & \
            (abs(events.PFCands.trkEta) <= 2.5) & \
            (abs(events.PFCands.dz) < 0.05) & \
            (abs(events.PFCands.d0) < 0.05) & \
            (events.PFCands.puppiWeight > 0.1)
            #(events.PFCands.dzErr < 0.05)
        
        cutTrackPF = (events.PFCands.pt >= 1) & (events.PFCands.charge == 0)

        IDCands = ak.packed(Cands[cutPF])
        NeutralCands = ak.packed(Cands[cutTrackPF])

        LostTracks = ak.zip({
            "pt": events.lostTracks.pt,
            "eta": events.lostTracks.eta,
            "phi": events.lostTracks.phi,
            "mass": 0.0,
            "pdgId": -99,
            "charge": events.lostTracks.charge,
            "inclusivePt": events.lostTracks.pt
        }, with_name="Momentum4D")

        cutLost = (events.lostTracks.fromPV > 1) & \
            (events.lostTracks.pt >= 1) & \
            (abs(events.lostTracks.eta) <= 2.5) & \
            (abs(events.lostTracks.dz) < 0.05) & \
            (abs(events.lostTracks.d0) < 0.05) & \
            (events.lostTracks.puppiWeight > 0.1)
            #(events.lostTracks.dzErr < 0.05)
        
        cutTrackLost = (events.lostTracks.pt >= 1) & (events.lostTracks.charge == 0) 

        lostIDCands = ak.packed(LostTracks[cutLost])
        LostNeutralCands = ak.packed(LostTracks[cutTrackLost])

        # dimensions of tracks = events x tracks in event x 4 momenta
        totalTracks = ak.concatenate([IDCands, lostIDCands], axis=1)
        totalNeutralTracks = ak.concatenate([NeutralCands, LostNeutralCands], axis=1)

        # Sorting out the tracks that overlap with VBF jets
        totalTracks = totalTracks[(totalTracks.deltaR(VBFCands.jet1) >= 0.4) & (totalTracks.deltaR(VBFCands.jet2) >= 0.4)]
        totalNeutralTracks = totalNeutralTracks[(totalNeutralTracks.deltaR(VBFCands.jet1) >= 0.4) & (totalNeutralTracks.deltaR(VBFCands.jet2) >= 0.4)]
        
        return events, totalTracks, totalNeutralTracks

    def clusterizeTracks(self, events, tracks):
        # anti-kt, dR=1.5 jets
        nSmallEvents = 5000
        smallEvents = len(events) < nSmallEvents
        if not smallEvents:
          jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)        
          cluster = fastjet.ClusterSequence(tracks, jetdef)
          ak15_jets   = ak.with_name(cluster.inclusive_jets(min_pt=0),"Momentum4D") # These are the ak15_jets
          ak15_consts = ak.with_name(cluster.constituents(min_pt=0),"Momentum4D")   # And these are the collections of constituents of the ak15_jets
          clidx = cluster.constituent_index()
          return events, ak15_jets, ak15_consts, clidx
        else: #With few events/file the thing crashes because of FastJet so we are going to create "fake" events
          ncopies     = round(nSmallEvents/(len(events)))
          oldtracks   = tracks
          for i in range(ncopies):
            tracks = ak.concatenate([tracks, oldtracks], axis=0) # I.e. duplicate our events until we are feeding 1000 events to the clusterizer
          jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)
          cluster = fastjet.ClusterSequence(tracks, jetdef)
          ak15_jets   = ak.with_name(cluster.inclusive_jets(min_pt=0),"Momentum4D") # These are the ak15_jets
          ak15_consts = ak.with_name(cluster.constituents(min_pt=0),"Momentum4D")   # And these are the collections of constituents of the ak15_jets
          clidx = cluster.constituent_index()
          # But now we have to delete the repeated set of events
          return events, ak15_jets[:len(oldtracks)], ak15_consts[:len(oldtracks)], clidx[:len(oldtracks)]

    def lundJetDeclustering(self, jet):
        declusterings = []
        currentJet = jet

        while True:
            parent1 = fastjet.PseudoJet()
            parent2 = fastjet.PseudoJet()
            hasParents = currentJet.has_parents(parent1, parent2)
            if not hasParents:
                break

            if parent1.pt() < parent2.pt():
                parent1, parent2 = parent2, parent1

            deltaR = parent1.delta_R(parent2)
            lnInvDelta = np.log(1.0 / deltaR)

            kt = parent2.pt() * deltaR
            lnKt = np.log(kt)

            logm = np.log((parent1 + parent2).m())
            z = parent2.pt() / (parent1.pt() + parent2.pt())
            logz = np.log(z)
            logkappa = np.log(z * deltaR)

            deltaEta = parent2.eta() - parent1.eta()
            deltaPhi = np.arccos(np.cos(parent2.phi() - parent1.phi()))
            psi = np.arctan2(deltaEta, deltaPhi)

            declusterings.append({
                'lnInvDelta': lnInvDelta,
                'lnKt': lnKt,
                'logm': logm,
                'logz': logz,
                'logkappa': logkappa,
                'psi': psi
            })
            currentJet = parent1

        declusteringsSecondary = []
        currentJet = jet

        while True:
            parent1 = fastjet.PseudoJet()
            parent2 = fastjet.PseudoJet()
            hasParents = currentJet.has_parents(parent1, parent2)
            if not hasParents:
                break

            if parent1.pt() < parent2.pt():
                parent1, parent2 = parent2, parent1

            deltaR = parent1.delta_R(parent2)
            lnInvDelta = np.log(1.0 / deltaR)

            kt = parent2.pt() * deltaR
            lnKt = np.log(kt)

            logm = np.log((parent1 + parent2).m())
            z = parent2.pt() / (parent1.pt() + parent2.pt())
            logz = np.log(z)
            logkappa = np.log(z * deltaR)

            deltaEta = parent2.eta() - parent1.eta()
            deltaPhi = np.arccos(np.cos(parent2.phi() - parent1.phi()))
            psi = np.arctan2(deltaEta, deltaPhi)

            declusteringsSecondary.append({
                'lnInvDelta': lnInvDelta,
                'lnKt': lnKt,
                'logm': logm,
                'logz': logz,
                'logkappa': logkappa,
                'psi': psi
            })
            currentJet = parent2
        return declusterings, declusteringsSecondary

    def lundDeclusterizer(self, constituents):
        lundPlanes = []
        secondaryLundPlanes = []
        lead_constituents = constituents[:, 0]

        for event_constituents in lead_constituents:
            if len(event_constituents) == 0:
                lundPlanes.append([])
                secondaryLundPlanes.append([])
                continue

            pseudojets = []
            for p in event_constituents:
                pj = fastjet.PseudoJet(p.px, p.py, p.pz, p.energy)
                pseudojets.append(pj)

            ca_jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 1.5)
            cluster_ca = fastjet.ClusterSequence(pseudojets, ca_jetdef)
            ca_jets = cluster_ca.inclusive_jets(0.0)

            if len(ca_jets) == 0:
                lundPlanes.append([])
                secondaryLundPlanes.append([])
                continue

            ca_jets_sorted = sorted(ca_jets, key=lambda jet: jet.pt(), reverse=True)
            ca_jet = ca_jets_sorted[0]

            declusterings, secondaryDeclusterings = self.lundJetDeclustering(ca_jet)
            lundPlanes.append(declusterings)
            secondaryLundPlanes.append(secondaryDeclusterings)

        return ak.Array(lundPlanes), ak.Array(secondaryLundPlanes)

    def shouldContinueAfterCut(self, events, out):
        #if debug: print("Conversion to pandas...")
        if len(events) == 0:
            outdfs  = []
            outcols = []
            print("No events pass cut, stopping...")
            for channel in out.keys():
                outcols.append(channel)
                if len(out[channel][0]) == 0:
                    print("Create empty frame") 
                    outdfs = pd.DataFrame(['empty'], columns=['empty'])
                else:              
                    if self.isMC:
                        out[channel][0]["genweight"] = out[channel][1].genWeight[:]

                if not isinstance(out[channel][0], pd.DataFrame): 
                   out[channel][0] = self.ak_to_pandas(out[channel][0])
            self.save_dfs([out[key][0] for key in out], [key for key in out], self.chunkTag)

            return False
        else: 
            return True

    def applyCutToAllCollections(self, cut): # Cut has to by a selection applicable across all collections, i.e. something defined per event
            self.events    = self.events[cut]
            self.jets      = self.jets[cut]
            self.VBFCands  = self.VBFCands[cut]
            self.tracks    = self.tracks[cut]
            self.neutralTracks = self.neutralTracks[cut]
            self.clusters  = self.clusters[cut]
            self.constituents  = self.constituents[cut]

    def process(self, events):
        np.random.seed(max(0,min(events.event[0], 2**31))) # This ensures reproducibility of results (i.e. for the random track dropping), while also getting different random numbers per file to avoid biases (like always dropping the first track, etc.
        debug    = True  # If we want some prints in the middle
        self.chunkTag = "out_%i_%i_%i.hdf5"%(events.event[0], events.luminosityBlock[0], events.run[0]) #Unique tag to get different outputs per tag
        fullFile = self.output_location + "/" + self.chunkTag
        print("Check file %s"%fullFile)
        if os.path.isfile(fullFile): 
            print("SKIP")
            return self.accumulator.identity()

        self.doTracks   = True  # Make it false, and it will speed things up but not run the tracks
        self.doClusters = True  # Make it false, and it will speed things up but not run the clusters
        self.doGen      = False if not(self.isDY) else True # In case we want info on the gen level, we do need it for the buggy DY samples (to get proper stitching)

        # Main processor code
        # ------------------------------------------------------------------------------------
        # ------------------------------- DEFINE OUTPUTS -------------------------------------
        # ------------------------------------------------------------------------------------

        accumulator    = self.accumulator.identity()
        # Each track is one selection level
        outputs = {
            "onecluster"  :[{},[]],   # Has Two Leptons, pT and Trigger requirements
        }

        dataset = events.metadata['dataset']
        if self.isMC: self.gensumweight = ak.sum(events.genWeight)

        if not(self.isMC): doGen = False

        # ------------------------------------------------------------------------------------
        # ------------------------------- OBJECT LOADING -------------------------------------
        # ------------------------------------------------------------------------------------

        # Apply Trigger
        self.events = self.selectByTrigger(events)

        # Jet selection
        self.events, self.VBFCands, self.jets = self.selectByJets(events)

        self.events, self.tracks, self.neutralTracks = self.selectByTracks(self.events, self.VBFCands)

        self.events, self.clusters, self.constituents, clidx  = self.clusterizeTracks(self.events, self.tracks)

        highpt_clusters = ak.argsort(self.clusters.pt, axis=1, ascending=False, stable=True)
        self.clusters   = self.clusters[highpt_clusters]
        self.constituents = self.constituents[highpt_clusters]
        clidx = clidx[highpt_clusters]

        self.constituents = ak.zip(
            {
            "pt": self.constituents.pt,
            "eta": self.constituents.eta,
            "phi": self.constituents.phi,
            "mass": self.constituents.mass,
            "pdgId": ak.unflatten(self.tracks.pdgId[ak.flatten(clidx,axis=2)], ak.flatten(ak.num(self.constituents, axis=2)), axis=1),
            }, with_name="Momentum4D")
        
        cutOneCluster = (ak.num(self.clusters) != 0)
        self.applyCutToAllCollections(cutOneCluster)

        self.lundVariables, self.secondaryLundVariables = self.lundDeclusterizer(self.constituents)
        
        self.lnKt = self.lundVariables['lnKt']
        self.lnInvDelta = self.lundVariables['lnInvDelta']

        lundselection = (self.lnKt >= np.log(2)) & (self.lnInvDelta < 2.0)

        # Reading and calculating S/B likelihood
        dfRatio = pd.read_csv('/eos/user/g/gdecastr/SUEPCoffea_dask/VBF_Checks/Lund/Ratio_VBF_2_2_QCD.csv')
        xlim = [np.log(1/1.5), 5]
        ylim = [-2, 7]
        xedges = np.linspace(xlim[0], xlim[1], 26)  # ln_inv_Delta bin edges
        yedges = np.linspace(ylim[0], ylim[1], 26)  # ln_kt bin edges
        rhoRatioFlat = dfRatio['rho_ratio'].values
        rhoRatio = rhoRatioFlat.reshape((len(xedges)-1, len(yedges)-1))

        lnInvDeltaFlat = ak.to_numpy(ak.flatten(self.lnInvDelta))
        lnKtFlat = ak.to_numpy(ak.flatten(self.lnKt))

        xIndicesFlat = np.searchsorted(xedges, lnInvDeltaFlat, side='right') - 1
        yIndicesFlat = np.searchsorted(yedges, lnKtFlat, side='right') - 1
        xIndicesFlat = np.clip(xIndicesFlat, 0, len(xedges) - 2)
        yIndicesFlat = np.clip(yIndicesFlat, 0, len(yedges) - 2)

        rhoValuesFlat = rhoRatio[xIndicesFlat, yIndicesFlat]

        counts = ak.num(self.lnKt, axis=1)
        rhoValues = ak.unflatten(rhoValuesFlat, counts)

        nEmissions = ak.num(self.lnKt, axis=1)

        # Outputted Lund variables
        self.numSelectedEmissions = ak.sum(lundselection, axis=1)
        self.numEmissions = nEmissions
        self.emissionSelectionProportion = ak.where(self.numEmissions > 0, self.numSelectedEmissions / self.numEmissions, 0)
        self.rhoProduct = ak.prod(rhoValues, axis=1)
        self.nthRhoProduct = ak.where(nEmissions > 0, self.rhoProduct ** (1.0 / nEmissions), 0.0)

        self.varsToDo = [""] 
        self.var = "" # To keep track of the systematic variation: "" == nominal
        for var in self.varsToDo:
            self.var = var
            outputs["onecluster"+var] = [self.doAllPlots("onecluster"+var, debug), self.events]
            if not(self.shouldContinueAfterCut(self.events, outputs)): continue
            if debug: print("%i events pass onecluster cuts. Doing more stuff..."%len(self.events))
        # ------------------------------------------------------------------------------
        # -------------------------------- SAVING --------------------------------------
        # ------------------------------------------------------------------------------

        for out in outputs:
            if debug: print("Conversion to pandas...")
            if not isinstance(outputs[out][0], pd.DataFrame):
                if debug: print("......%s"%out)
                outputs[out][0] = self.ak_to_pandas(outputs[out][0])

        if debug: print("DFS saving....")

        self.save_dfs([outputs[key][0] for key in outputs], [key for key in outputs], self.chunkTag)

        return accumulator   

    def doAllPlots(self, channel, debug=True):
        # ------------------------------------------------------------------------------
        # ------------------------------- PLOTTING -------------------------------------
        # ------------------------------------------------------------------------------
        out = {}
        # Define outputs for plotting
        if debug: print("Saving reco variables for channel %s"%channel)

        out["H_T"]         = ak.sum(self.jets.pt, axis=1)[:]
        out["Ak4SUEPOverlap"] = ak.sum(self.jets.deltaR(self.clusters[:, 0]) < 1.5, axis=1)
        out["nJets"] = ak.num(self.jets, axis=1)
        out["MET_pt"] = self.events.PuppiMET.pt[:]
        out["nBLoose"]        = ak.sum((self.jets.btag >= 0.0490), axis=1)[:]
        out["nBMedium"]       = ak.sum((self.jets.btag >= 0.2783), axis=1)[:]
        out["nBTight"]        = ak.sum((self.jets.btag >= 0.7100), axis=1)[:]

        is_jet1_leading = self.VBFCands.jet1.pt > self.VBFCands.jet2.pt
        
        # Lead jet properties
        out["leadJetPt"]  = ak.where(is_jet1_leading, self.VBFCands.jet1.pt, self.VBFCands.jet2.pt)
        out["leadJetEta"] = ak.where(is_jet1_leading, self.VBFCands.jet1.eta, self.VBFCands.jet2.eta)
        out["leadJetPhi"] = ak.where(is_jet1_leading, self.VBFCands.jet1.phi, self.VBFCands.jet2.phi)
        out["leadJetMass"] = ak.where(is_jet1_leading, self.VBFCands.jet1.mass, self.VBFCands.jet2.mass)

        out["leadJetchHEF"] = ak.where(is_jet1_leading, self.VBFCands.jet1.chHEF, self.VBFCands.jet2.chHEF)
        out["leadJetneHEF"] = ak.where(is_jet1_leading, self.VBFCands.jet1.neHEF, self.VBFCands.jet2.neHEF)
        out["leadJetchEmEF"] = ak.where(is_jet1_leading, self.VBFCands.jet1.chEmEF, self.VBFCands.jet2.chEmEF)
        out["leadJetneEmEF"] = ak.where(is_jet1_leading, self.VBFCands.jet1.neEmEF, self.VBFCands.jet2.neEmEF)
        out["leadJetmuEF"] = ak.where(is_jet1_leading, self.VBFCands.jet1.muEF, self.VBFCands.jet2.muEF)
        out["leadJetChFraction"] = ak.where(is_jet1_leading, self.VBFCands.jet1.chMultiplicity / self.VBFCands.jet1.nConstituents, self.VBFCands.jet2.chMultiplicity / self.VBFCands.jet2.nConstituents)
        out["leadJetNeFraction"] = ak.where(is_jet1_leading, self.VBFCands.jet1.neMultiplicity / self.VBFCands.jet1.nConstituents, self.VBFCands.jet2.neMultiplicity / self.VBFCands.jet2.nConstituents)


        # Sublead jet properties
        out["subleadJetPt"]  = ak.where(is_jet1_leading, self.VBFCands.jet2.pt, self.VBFCands.jet1.pt)
        out["subleadJetEta"] = ak.where(is_jet1_leading, self.VBFCands.jet2.eta, self.VBFCands.jet1.eta)
        out["subleadJetPhi"] = ak.where(is_jet1_leading, self.VBFCands.jet2.phi, self.VBFCands.jet1.phi)
        out["subleadJetMass"] = ak.where(is_jet1_leading, self.VBFCands.jet2.mass, self.VBFCands.jet1.mass)

        out["subleadJetchHEF"] = ak.where(is_jet1_leading, self.VBFCands.jet2.chHEF, self.VBFCands.jet1.chHEF)
        out["subleadJetneHEF"] = ak.where(is_jet1_leading, self.VBFCands.jet2.neHEF, self.VBFCands.jet1.neHEF)
        out["subleadJetchEmEF"] = ak.where(is_jet1_leading, self.VBFCands.jet2.chEmEF, self.VBFCands.jet1.chEmEF)
        out["subleadJetneEmEF"] = ak.where(is_jet1_leading, self.VBFCands.jet2.neEmEF, self.VBFCands.jet1.neEmEF)
        out["subleadJetmuEF"] = ak.where(is_jet1_leading, self.VBFCands.jet2.muEF, self.VBFCands.jet1.muEF)
        out["subleadJetChFraction"] = ak.where(is_jet1_leading, self.VBFCands.jet2.chMultiplicity / self.VBFCands.jet2.nConstituents, self.VBFCands.jet1.chMultiplicity / self.VBFCands.jet1.nConstituents)
        out["subleadJetNeFraction"] = ak.where(is_jet1_leading, self.VBFCands.jet2.neMultiplicity / self.VBFCands.jet2.nConstituents, self.VBFCands.jet1.neMultiplicity / self.VBFCands.jet1.nConstituents)

        out["ptjj"] = (self.VBFCands.jet1 + self.VBFCands.jet2).pt
        out["etajj"] = (self.VBFCands.jet1 + self.VBFCands.jet2).eta
        out["phijj"] = (self.VBFCands.jet1 + self.VBFCands.jet2).phi
        out["mjj"] = (self.VBFCands.jet1 + self.VBFCands.jet2).mass

        out["deta"] = abs(self.VBFCands.jet1.eta - self.VBFCands.jet2.eta)
        out["dphi"] = np.arccos(np.cos(self.VBFCands.jet1.phi - self.VBFCands.jet2.phi))
        out["dR"] = self.VBFCands.jet1.deltaR(self.VBFCands.jet2)

        out["ntracks"]     = ak.num(self.tracks, axis=1)[:]

        out["leadcluster_pt"]      = self.clusters.pt[:,0]
        out["leadcluster_eta"]     = self.clusters.eta[:,0]
        out["leadcluster_phi"]     = self.clusters.phi[:,0]
        out["leadcluster_m"]     = self.clusters.mass[:,0]
        out["leadcluster_ntracks"] = ak.num(self.constituents[:,0], axis = 1)
        out["leadcluster_neutralFraction"] = ak.sum(self.neutralTracks.deltaR(self.clusters[:, 0]) < 1.5, axis=1) / (out["leadcluster_ntracks"] + ak.sum(self.neutralTracks.deltaR(self.clusters[:, 0]) < 1.5, axis=1))

        boost_leading = ak.zip({
            "px": self.clusters[:,0].px*-1,
            "py": self.clusters[:,0].py*-1,
            "pz": self.clusters[:,0].pz*-1,
            "mass": self.clusters[:,0].mass
        }, with_name="Momentum4D")

        leadingclustertracks = self.constituents[:,0]
        leadingclustertracks_boostedagainstSUEP   = leadingclustertracks.boost_p4(boost_leading)

        evalsL = self.sphericity(self.events, leadingclustertracks, 2) 
        evalsC = self.sphericity(self.events, leadingclustertracks_boostedagainstSUEP, 2)

        out["leadclusterSpher_L"] =  np.real(1.5*(evalsL[:,0] + evalsL[:,1]))
        out["leadclusterSpher_C"] =  np.real(1.5*(evalsC[:,0] + evalsC[:,1]))

        evalsL_r1 = self.sphericity(self.events, leadingclustertracks, 1) 
        evalsC_r1 = self.sphericity(self.events, leadingclustertracks_boostedagainstSUEP, 1)

        out["leadclusterSpher_L_r1"] =  np.real(1.5*(evalsL_r1[:,0] + evalsL_r1[:,1]))
        out["leadclusterSpher_C_r1"] =  np.real(1.5*(evalsC_r1[:,0] + evalsC_r1[:,1]))

        if self.isMC: out["genWeight"] = self.events.genWeight[:] 

        out['numSelectedEmissions'] = self.numSelectedEmissions
        out['numEmissions'] = self.numEmissions
        out['emissionSelectionProportion'] = self.emissionSelectionProportion
        out['rhoProduct'] = self.rhoProduct
        out['nthRhoProduct'] = self.nthRhoProduct

        
        lnInvDeltaList, lnKtList, lnMList, lnZList, lnKappaList, psiList, lnInvDeltaList_2, lnKtList_2, lnMList_2, lnZList_2, lnKappaList_2, psiList_2, genWeightList = [], [], [], [], [], [], [], [], [], [], [], [], []

        for idx, declusterings in enumerate(self.lundVariables):
            lnInvDeltaVals = [d['lnInvDelta'] for d in declusterings]
            lnKtVals = [d['lnKt'] for d in declusterings]
            lnMVals = [d['logm'] for d in declusterings]
            lnZVals = [d['logz'] for d in declusterings]
            lnKappaVals = [d['logkappa'] for d in declusterings]
            psiVals = [d['psi'] for d in declusterings]
            
            lnInvDeltaList.append(lnInvDeltaVals)
            lnKtList.append(lnKtVals)
            lnMList.append(lnMVals)
            lnZList.append(lnZVals)
            lnKappaList.append(lnKappaVals)
            psiList.append(psiVals)

            lnInvDeltaVals_2 = [d['lnInvDelta'] for d in self.secondaryLundVariables[idx]]
            lnKtVals_2 = [d['lnKt'] for d in self.secondaryLundVariables[idx]]
            lnMVals_2 = [d['logm'] for d in self.secondaryLundVariables[idx]]
            lnZVals_2 = [d['logz'] for d in self.secondaryLundVariables[idx]]
            lnKappaVals_2 = [d['logkappa'] for d in self.secondaryLundVariables[idx]]
            psiVals_2 = [d['psi'] for d in self.secondaryLundVariables[idx]]
            
            lnInvDeltaList_2.append(lnInvDeltaVals_2)
            lnKtList_2.append(lnKtVals_2)
            lnMList_2.append(lnMVals_2)
            lnZList_2.append(lnZVals_2)
            lnKappaList_2.append(lnKappaVals_2)
            psiList_2.append(psiVals_2)

            if self.isMC: genWeightList.append(self.events.genWeight[idx])    

        # Create DataFrame
        if self.isMC:
            df = pd.DataFrame({
                'lnInvDelta': lnInvDeltaList,
                'lnKt': lnKtList,
                'logm': lnMList,
                'logz': lnZList,
                'logkappa': lnKappaList,
                'psi': psiList,
                'lnInvDelta_2': lnInvDeltaList_2,
                'lnKt_2': lnKtList_2,
                'logm_2': lnMList_2,
                'logz_2': lnZList_2,
                'logkappa_2': lnKappaList_2,
                'psi_2': psiList_2,
                'genWeight': genWeightList
            })
        else:
            df = pd.DataFrame({
                'lnInvDelta': lnInvDeltaList,
                'lnKt': lnKtList,
                'logm': lnMList,
                'logz': lnZList,
                'logkappa': lnKappaList,
                'psi': psiList,
                'lnInvDelta_2': lnInvDeltaList_2,
                'lnKt_2': lnKtList_2,
                'logm_2': lnMList_2,
                'logz_2': lnZList_2,
                'logkappa_2': lnKappaList_2,
                'psi_2': psiList_2
            })

        # Save to CSV
        df.to_csv(self.fileName, index=False)

        return out

    def postprocess(self, accumulator):
        return accumulator
 
