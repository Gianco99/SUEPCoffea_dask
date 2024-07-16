import os
import ROOT

def getSignalStrength(f):
    rF = ROOT.TFile(f, "READ")
    tree = rF.Get("tree_fit_sb;1")
    signal_strength = "999."

    if tree:
        rBranch = tree.GetBranch("r")
        if rBranch:
            for ev in tree:
                signal_strength = str(ev.r)
                break  # Assuming we want the first entry only

    rF.Close()
    return signal_strength

output_file_path = "hadronic_fit_SR.txt"
with open(output_file_path, "w") as out_file:
    out_file.write("# mS : mD : T : r \n")

    for fil in os.listdir("./"):
        if "fitDiagnosticsCombinedSR_" not in fil:
            continue

        print(fil)
        higgs, SUEP, mode, mass, mDark, T = fil.split("_")
        mass = mass.replace("mS", "")
        mDark = mDark.replace("mD", "")
        T = ".".join(T.split(".")[0:2]).replace("T", "")

        signal_strength = getSignalStrength(fil)
        out_file.write(f"{mass} : {mDark} : {T} : {signal_strength}\n")
