import os
import argparse

def check_file(fi):
    if os.path.isfile(fi) and os.path.getsize(fi) > 6500:
        return True
    return False

def main(directory, rMin, rMax, points):
    iJ = 0
    all_files = os.listdir(directory)

    for f in all_files:
        if "Combined" not in f or "txt" not in f or "higgsCombine" in f:
            continue
        print(f)
        froot = f.replace(".txt", "")
        if check_file(os.path.join(directory, f"higgsCombine{froot}.MultiDimFit.mH120.root")):
            continue
        print("Pass")
        iJ += 1
        job_file_path = os.path.join("exec", f"job_{iJ}.sh")
        with open(job_file_path, "w") as fil:
            fil.write("#!/bin/csh\n")
            fil.write("cd /eos/user/g/gdecastr/CMSSW_11_3_4/src/\n")
            fil.write("source /cvmfs/cms.cern.ch/cmsset_default.csh\n")
            fil.write("cmsenv\n")
            fil.write(f"cd {directory}\n")
            fil.write("ulimit -s unlimited\n")
            fil.write(f"text2workspace.py {f}\n")
            if not check_file(os.path.join(directory, f"higgsCombine{froot}.MultiDimFit.mH120.root")):
                fil.write(f"combine -M MultiDimFit {f} --algo grid --saveNLL --cminDefaultMinimizerStrategy 0 --setParameterRanges rCRTT16=0,2:rCRTT16APV=0,2:rCRTT17=0,2:rCRTT18=0,2:rCRDY16=0,2:rCRDY16APV=0,2:rCRDY17=0,2:rCRDY18=0,2:rSR16=0,2:rSR16APV=0,2:rSR17=0,2:rSR18=0,2 --forceRecreateNLL --rMin={rMin} --rMax={rMax} --points={points} -n {froot}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--directory", type=str, required=True, help="Directory to process")
    parser.add_argument("--rMin", type=int, required=True, help="Minimum value of r")
    parser.add_argument("--rMax", type=int, required=True, help="Maximum value of r")
    parser.add_argument("--points", type=int, required=True, help="Number of points")

    args = parser.parse_args()
    main(args.directory, args.rMin, args.rMax, args.points)