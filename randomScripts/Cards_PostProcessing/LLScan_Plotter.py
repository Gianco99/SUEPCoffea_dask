import os
import argparse
import uproot
import matplotlib.pyplot as plt

def plot_likelihood_scan(root_filename, color, label):
    root_file = uproot.open(root_filename)
    tree = root_file["limit;1"]
    nll_values = tree["deltaNLL"].array()
    r_values = tree["r"].array()
    plt.scatter(r_values, 2*nll_values, marker='.', c=color, label=label)

def find_files(directory, md_value=None, t_value=None):
    files = [f for f in os.listdir(directory) if "SRCR" in f and "MultiDimFit" in f and f.endswith(".root")]
    if md_value is not None and t_value is not None:
        md_t_substr = f"mD{md_value}_T{t_value}"
        files = [f for f in files if md_t_substr in f]
    return [f for f in files if os.path.getsize(os.path.join(directory, f)) > 6500]

def create_plots(input_dir, output_dir, md_value=None, t_value=None):
    srcr_files = find_files(input_dir, md_value, t_value)

    for srcr_file in srcr_files:
        srcr_path = os.path.join(input_dir, srcr_file)
        sr_file = os.path.join(input_dir, srcr_file.replace("SRCR", "SR"))
        crdy_file = os.path.join(input_dir, srcr_file.replace("SRCR", "CRDY"))
        crtt_file = os.path.join(input_dir, srcr_file.replace("SRCR", "CRTT"))

        existing_files = [(sr_file, 'c', 'SR'), (crdy_file, 'b', 'CRDY'), (crtt_file, 'g', 'CRTT'), (srcr_path, 'r', 'SRCR')]
        existing_files = [f for f in existing_files if os.path.exists(f[0]) and os.path.getsize(f[0]) > 6500]

        if existing_files:
            plt.figure(figsize=(8, 6))
            for file, color, label in existing_files:
                plot_likelihood_scan(file, color, label)

            plt.xlabel('r')
            plt.ylabel(r'2$\times$ $\Delta$NLL')
            plt.title(r'2D Plot of $\Delta$NLL vs r')
            plt.grid(True)
            plt.legend()
            plt.axhline(y=1, color='k', linestyle='--', label='y=1')
            plt.axhline(y=4, color='k', linestyle='--', label='y=4')
            
            # Save plot with y-lim (0, 20)
            plt.ylim(0, 20)
            output_filename = os.path.join(output_dir, f"{os.path.splitext(srcr_file)[0]}_ylim20.png")
            plt.savefig(output_filename)
            plt.close()
            
            # Save plot with y-lim (0, 200)
            plt.figure(figsize=(8, 6))
            for file, color, label in existing_files:
                plot_likelihood_scan(file, color, label)

            plt.xlabel('r')
            plt.ylabel(r'2$\times$ $\Delta$NLL')
            plt.title(r'2D Plot of $\Delta$NLL vs r')
            plt.grid(True)
            plt.legend()
            plt.axhline(y=1, color='k', linestyle='--', label='y=1')
            plt.axhline(y=4, color='k', linestyle='--', label='y=4')
            plt.ylim(0, 200)
            output_filename = os.path.join(output_dir, f"{os.path.splitext(srcr_file)[0]}_ylim200.png")
            plt.savefig(output_filename)
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot likelihood scans from ROOT files.")
    parser.add_argument("input_dir", help="Input directory containing ROOT files")
    parser.add_argument("output_dir", help="Output directory to save plots")
    parser.add_argument("--mD", type=float, help="Specify mD value (as written in file) to filter files")
    parser.add_argument("--T", type=float, help="Specify T value (as written in file) to filter files")
    
    args = parser.parse_args()
    
    create_plots(args.input_dir, args.output_dir, args.mD, args.T)
