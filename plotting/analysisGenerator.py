import sys
import argparse

def getYears(choice):
    validYears = {
        '18': 'UL18',
        '17': 'UL17',
        '16': 'UL16',
        '16APV': 'UL16APV',
        'RunII': 'RunII'
    }
    if choice == 'toLoad':
        return validYears
    else:
        return {k: v for k, v in validYears.items() if k in ['18', '17', '16', '16APV']}

def getSampleMap():
    return {
        '1': "ZH/samples_withSF_nocuts_XXXX.py"
    }

def getPlotMap(choice):
    plotMap = {
        '1D': {
            'SR': 'ZH/plots_SR.py',
            'CRDY': 'ZH/plots_CRDY.py',
            'CRTT': 'ZH/plots_CRTT.py'
        },
        '2D': {
            'SR': 'ZH/plots_SR_2D.py',
            'CRDY': 'ZH/plots_CRDY_2D.py',
            'CRTT': 'ZH/plots_CRTT_2D.py'
        }
    }
    return plotMap

def getSystMap():
    return {
        '1': "--systFile ZH/systs_fullcorr_MC.py",
        '2': "--systFile ZH/systs_uncorrClosure.py"
    }

lumiMap = {
    'UL18': '59.9',
    'UL17': '41.6',
    'UL16': '16.4',
    'UL16APV': '19.9',
    'RunII': '137.0'
}

def generateBashScript(args):
    years = getYears(args.choice)
    
    if args.years:
        try:
            yearsToUse = [years[y] for y in args.years]
        except KeyError as e:
            print(f"Error: Invalid year choice {e.args[0]}")
            sys.exit(1)
    else:
        yearsToUse = list(years.values())
    
    if 'RunII' in yearsToUse and args.choice in ['toSave', 'cards']:
        print("Error: RunII is not available for toSave or cards.")
        sys.exit(1)

    sampleMap = getSampleMap()
    plotMap = getPlotMap(args.choice)
    systMap = getSystMap()

    if args.dim == '2D' and args.choice == 'cards':
        print("Error: 2D dimension is not available for cards.")
        sys.exit(1)

    if args.plotAll and args.choice == 'cards':
        print("Error: --plotAll is not available for cards.")
        sys.exit(1)

    if (args.doAll or args.manualMCStats or args.floatB or args.integrateBins) and args.choice != 'cards':
        print("Error: --doAll, --manualMCStats, --floatB, and --integrateBins are only available for cards.")
        sys.exit(1)

    if args.jobSubmit and args.choice != 'toSave':
        print("Error: --jobSubmit is only available for toSave.")
        sys.exit(1)

    if 'XXXX' not in args.outPath:
        print("Error: outPath must contain 'XXXX' as a placeholder for the year.")
        sys.exit(1)
    
    if 'XXXX' not in args.inPath:
        print("Error: inPath must contain 'XXXX' as a placeholder for the year.")
        sys.exit(1)

    sample = sampleMap[args.samplesFile]
    selection = plotMap[args.dim][args.plotsFile]
    systs = '' if args.dim == '2D' else systMap.get(args.syst, '')
    outPath = args.outPath
    plotAll = '--plotAll' if args.plotAll else ''
    extraCmds = ' '.join(args.extraCmds) if args.extraCmds else ''
    jobSubmitCmd = '--jobName {outPath.replace("XXXX", year)}_Jobs --batchsize 100 --resubmit --queue workday' if args.jobSubmit else ''

    if args.choice == 'toSave':
        commands = [
            f"python plotter_vh.py {sample.replace('XXXX', year)} {selection} -l {lumiMap[year]} {systs} --toSave {outPath.replace('XXXX', year)} {plotAll} {jobSubmitCmd} {extraCmds}"
            for year in yearsToUse
        ]

    elif args.choice == 'toLoad':
        inPath = args.inPath
        commands = [
            f"python plotter_vh.py {sample.replace('XXXX', year)} {selection} -l {lumiMap[year]} {systs} --toLoad {inPath.replace('XXXX', year)} --plotdir {outPath.replace('XXXX', year)} {plotAll} {extraCmds}"
            for year in yearsToUse
        ]

    elif args.choice == 'cards':
        inPath = args.inPath
        doAll = '--doAll' if args.doAll else ''
        manualMCStats = '--manualMCStats' if args.manualMCStats else ''
        floatB = '--floatB' if args.floatB else ''
        integrateBins = f'--integrateBins {args.integrateBins}' if args.integrateBins else ''
        
        commands = [
            f"python datacardMaker.py {sample.replace('XXXX', year)} {systs} {outPath.replace('XXXX', year)}/{year.replace('UL', '20')} --rootFile {inPath.replace('XXXX', year)}/leadclustertracks --var leadclustertracks --ABCD --year {year.replace('UL', '20')} --region {args.plotsFile} {doAll} {manualMCStats} {floatB} {integrateBins} {extraCmds}"
            for year in yearsToUse
        ]

    for command in commands:
        print(command + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bash scripts for ZH SUEP tasks")

    parser.add_argument('choice', choices=['toSave', 'toLoad', 'cards'], help='Task choice: toSave - Saving Histograms, toLoad - Plotting Histograms, cards - Running Cards')
    parser.add_argument('--years', nargs='+', choices=['18', '17', '16', '16APV', 'RunII'], help='Year choices: 18 - UL18, 17 - UL17, 16 - UL16, 16APV - UL16APV, RunII - RunII')
    parser.add_argument('--samplesFile', choices=['1'], default='1', help='Sample file choice: 1 - High statistics signals, standard selection')
    parser.add_argument('--dim', choices=['1D', '2D'], default='1D', help='Dimension choice: 1D or 2D')
    parser.add_argument('--plotsFile', choices=['SR', 'CRDY', 'CRTT'], default='SR', help='Plots file choice: SR, CRDY, CRTT')
    parser.add_argument('--syst', choices=['1', '2'], help='Systematics choice: 1 - Standard systematics, 2 - Standard systematics w/ fully decorrelated closure syst')
    parser.add_argument('--outPath', required=True, help='Output path with XXXX as year placeholder')
    parser.add_argument('--inPath', required=True, help='Path to histograms with XXXX as year placeholder')
    parser.add_argument('--extraCmds', nargs=argparse.REMAINDER, help='Additional commands to append')
    parser.add_argument('--plotAll', action='store_true', help='Plot all signal samples (only for toSave and toLoad)')
    parser.add_argument('--doAll', action='store_true', help='Make cards for all signal samples (only for cards)')
    parser.add_argument('--manualMCStats', action='store_true', help='Turn on ManualMCStats (only for cards)')
    parser.add_argument('--floatB', action='store_true', help='Float the background (only for cards)')
    parser.add_argument('--integrateBins', type=int, help='Turn on integrateBins with the specified value (only for cards)')
    parser.add_argument('--jobSubmit', action='store_true', help='Submit jobs (only for toSave)')

    args = parser.parse_args()
    generateBashScript(args)
