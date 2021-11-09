import os
import json
import argparse
#import uproot3 as uproot

#Import coffea specific features
from coffea.processor import run_uproot_job, futures_executor

#SUEP Repo Specific
from SUEP_coffea import *
from SumWeights import *

#Begin argparse
parser = argparse.ArgumentParser("")
parser.add_argument('--isMC', type=int, default=1, help="")
parser.add_argument('--jobNum', type=int, default=1, help="")
parser.add_argument('--era', type=str, default="2018", help="")
parser.add_argument('--doSyst', type=int, default=1, help="")
parser.add_argument('--infile', type=str, default=None, help="")
parser.add_argument('--dataset', type=str, default="X", help="")
parser.add_argument('--nevt', type=str, default=-1, help="")
options = parser.parse_args()

#Set up cross section for MC normalizations
xsection = 1.0
if options.isMC:
   with open(os.path.dirname(__file__) +'xsections_{}.json'.format(options.era)) as file:
       MC_xsecs = json.load(file)
   try:
       xsection *= MC_xsecs[options.dataset]["xsec"]
       xsection *= MC_xsecs[options.dataset]["kr"]
       xsection *= MC_xsecs[options.dataset]["br"]
   except:
       print("WARNING: I did not find the xsection for that MC sample. Check the dataset name and the relevant yaml file")

#Take care of the normalization factor. Returns a float that is the xsec/gensumweight. Branching fractions and kfactors are included(see above).
modules_gensum = []
if options.isMC:
    modules_gensum.append(XsecSumWeight(isMC=options.isMC, xsec = xsection,  era=int(options.era), do_syst=1, syst_var='', sample=options.dataset))

    for instance in modules_gensum:
        output = run_uproot_job(
            {instance.sample: [options.infile]},
            treename='Runs',
            processor_instance=instance,
            executor=futures_executor,
            executor_args={'workers': 10,
                           'schema': processor.NanoAODSchema,
                           'xrootdtimeout': 10,
            },
            chunksize=500000000
        )
        xsection = output

out_dir = os.getcwd()
modules_era = []
#Run the SUEP code. Note the xsection as input. For Data the xsection = 1.0 from above
modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1, xsec = xsection,  syst_var='', sample=options.dataset, weight_syst='' , flag=False, output_location=out_dir))
    

for instance in modules_era:
    output = run_uproot_job(
        {instance.sample: [options.infile]},
        treename='Events',
        processor_instance=instance,
        executor=futures_executor,
        executor_args={'workers': 10,
                       'schema': processor.NanoAODSchema,
                       'xrootdtimeout': 10,
        },
        chunksize=500000
    )