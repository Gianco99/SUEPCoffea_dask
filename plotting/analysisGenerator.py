import sys

def whatYear(choice):
    print("\nPlease choose the year(s) you wish to produce:\n")
    print("1. UL18")
    print("2. UL17")
    print("3. UL16")
    print("4. UL16APV")
    if int(choice) == 2:
        print("5. RunII")

    year_choice = input("\nEnter the number corresponding to the year you want to analyze (comma-separated for multiple choices): ")
    years = year_choice.split(',')

    valid_years = ['1', '2', '3', '4', '5']

    if int(choice) == 2:
        year_map = {
            '1': 'UL18',
            '2': 'UL17',
            '3': 'UL16',
            '4': 'UL16APV',
            '5': 'RunII'
        }
        valid_years = ['1', '2', '3', '4', '5']
    else:
        year_map = {
            '1': 'UL18',
            '2': 'UL17',
            '3': 'UL16',
            '4': 'UL16APV',
        }
        valid_years = ['1', '2', '3', '4']

    years_desired = []
    for year in years:
        year = year.strip()
        if year not in valid_years:
            print(f"Error: Invalid choice '{year}' is not recognized.")
            sys.exit(1)
        years_desired.append(year_map[year])
    
    return years_desired

def whatSample():
    print("\nPlease choose the type of samples you wish to use:\n")

    print("1. High statistics signals, standard selection")

    sample_choice = input("\nEnter the number corresponding to the type of samples you wish to use:\n")

    samples_map = {
        '1': "ZH/samples_withSF_nocuts_XXXX.py",
    }
    valid_samples = ['1']

    if sample_choice not in valid_samples:
        print(f"Error: Invalid choice '{sample_choice}' is not recognized.")
        sys.exit(1)
    
    return samples_map[sample_choice]

def whatPlots(choice):
    if int(choice) == 1:
        print("\nPlease choose whether you want 1D or 2D histograms:\n")
    else:
        print("\nPlease choose whether you want 1D or 2D plots:\n")

    print("1. 1D")
    print("2. 2D")

    dim_map = {
        '1': '1D',
        '2': '2D'
    }

    valid_dims = ['1', '2']
            
    dim_choice = input("\nEnter the number corresponding to the dimensionality you desire:\n")

    if dim_choice not in valid_dims:
        print(f"Error: Invalid choice '{dim_choice}' is not recognized.")
        sys.exit(1)

    dims = dim_map[dim_choice]

    print("\nPlease choose the baseline regional selection:\n")

    print("1. SR")
    print("2. CRDY")
    print("3. CRTT")

    selection_map = {
        '1': 'SR',
        '2': 'CRDY',
        '3': 'CRTT'
    }

    valid_sels = ['1', '2', '3']
            
    sel_choice = input("\nEnter the number corresponding to the selection you desire:\n")

    if sel_choice not in valid_sels:
        print(f"Error: Invalid choice '{sel_choice}' is not recognized.")
        sys.exit(1)

    selection = selection_map[sel_choice]

    if dims == '1D':
        if selection == 'SR' or selection == 'CRDY' or selection == 'CRTT':
            return [Nominal_1D(selection), '1D']
        elif selection == 'Unique':
            return Unique_1D()
    
    elif dims == '2D':
        if selection == 'SR' or selection == 'CRDY' or selection == 'CRTT':
            return [Nominal_2D(selection, choice), '2D']
        elif selection == 'Unique':
            return Unique_2D()

def Nominal_1D(selection):
    print("\nPlease select from the following 1D " + selection + " plots files:\n")

    print("1. Full kinematics, no dR selection, nominal ABCD bounds")
    print("2. Minimum for cards and fit, no dR selection, nominal ABCD bounds")
    print("3. Minimum for cards and fit, with dR selection, nominal ABCD bounds")
    print("4: Minimum for cards and fit, with dR selection, new ABCD bound")
    print("5: 'Cool' plots with dR selection and nominal ABCD bounds")
    print("6: 'Cool' plots with dR selection and new ABCD bounds")

    choice_map = {
        '1': 'ZH/plots_' + selection + '.py',
        '2': 'ZH/plots_forCards_' + selection + '.py',
        '3': 'ZH/plots_forCards_' + selection + '_dR.txt',
        '4': 'ZH/plots_forCards_' + selection + '_dR_CommonCut.py',
        '5': 'ZH/plots_' + selection + '_cool.py',
        '6': 'ZH/plots_' + selection + '_cool_NewABCD.py'
    }

    valid_choices = ['1', '2', '3', '4', '5', '6']

    choice = input("\nEnter the number corresponding to the selection you desire:\n")

    if choice not in valid_choices:
        print(f"Error: Invalid choice '{choice}' is not recognized.")
        sys.exit(1)
    
    return choice_map[choice] 

def Nominal_2D(selection, choice):
    print("\nPlease select from the following 2D " + selection + " plots files:\n")

    if selection == "SR":
        selection = "Default"

    if choice == '1':

        print("1. No dR selection")
        print("2. With dR selection")

        choice_map = {
            '1': 'ZH/plots_forCards_' + selection + '_2D.py',
            '2': 'ZH/plots_forCards_' + selection + '_2D_dR.py'
        }
        valid_choices = ['1', '2']

    elif choice == '2':

        print("1. No dR selection, nominal ABCD bounds")
        print("2. No dR selection, new ABCD bounds")
        print("3. With dR selection, nominal ABCD bounds")
        print("4. With dR selection, new ABCD bounds")

        choice_map = {
            '1': 'ZH/plots_forCards_' + selection + '_2D.py --addLines 20,0,20,1000 --addLines 10,0,10,1000 --addLines 0,100,100,100 --addLines 0,250,100,250 --addText A:60,20:0.05 --addText B1:60,120:0.05 --addText B2:60,270:0.05 --addText C1:12,20:0.05 --addText D1:12,120:0.05 --addText E1:12,270:0.05 --addText C2:3,20:0.05 --addText D2:3,120:0.05 --addText E2:3,270:0.05',
            '2': 'ZH/plots_forCards_' + selection + '_2D.py --addLines 21,0,21,1000 --addLines 14,0,14,1000 --addLines 0,135,100,135 --addLines 0,220,100,220 --addText A:60,45:0.05 --addText B1:60,145:0.05 --addText B2:60,270:0.05 --addText C1:15,45:0.05 --addText D1:15,145:0.05 --addText E1:15,270:0.05 --addText C2:5,45:0.05 --addText D2:5,145:0.05 --addText E2:5,270:0.05',
            '3': 'ZH/plots_forCards_' + selection + '_2D_dR.py --addLines 20,0,20,1000 --addLines 10,0,10,1000 --addLines 0,100,100,100 --addLines 0,250,100,250 --addText A:60,20:0.05 --addText B1:60,120:0.05 --addText B2:60,270:0.05 --addText C1:12,20:0.05 --addText D1:12,120:0.05 --addText E1:12,270:0.05 --addText C2:3,20:0.05 --addText D2:3,120:0.05 --addText E2:3,270:0.05',
            '4': 'ZH/plots_forCards_' + selection + '_2D_dR.py --addLines 21,0,21,1000 --addLines 14,0,14,1000 --addLines 0,135,100,135 --addLines 0,220,100,220 --addText A:60,45:0.05 --addText B1:60,145:0.05 --addText B2:60,270:0.05 --addText C1:15,45:0.05 --addText D1:15,145:0.05 --addText E1:15,270:0.05 --addText C2:5,45:0.05 --addText D2:5,145:0.05 --addText E2:5,270:0.05',
        }
        valid_choices = ['1', '2', '3', '4']

    choice = input("\nEnter the number corresponding to the selection you desire:\n")

    if choice not in valid_choices:
        print(f"Error: Invalid choice '{choice}' is not recognized.")
        sys.exit(1)
    
    return choice_map[choice] 

def whatCards():
    print("\nPlease choose the baseline regional selection:\n")

    print("1. SR")
    print("2. CRDY")
    print("3. CRTT")

    selection_map = {
        '1': 'SR',
        '2': 'CRDY',
        '3': 'CRTT'
    }

    valid_sels = ['1', '2', '3']
            
    sel_choice = input("\nEnter the number corresponding to the selection you desire:\n")

    if sel_choice not in valid_sels:
        print(f"Error: Invalid choice '{sel_choice}' is not recognized.")
        sys.exit(1)

    return selection_map[sel_choice] 

def whatSysts():
    print("\nPlease choose the type of systematics you wish to use:\n")

    print("1. Standard systematics")
    print("2. Standard systematics w/ fully decorrelated closure syst")

    syst_choice = input("\nEnter the number corresponding to the type of samples you wish to use:\n")

    syst_map = {
        '1': "--systFile ZH/systs_fullcorr_MC.py",
        '2': "--systFile ZH/systs_uncorrClosure.py"
    }
    valid_systs = ['1', '2']

    if syst_choice not in valid_systs:
        print(f"Error: Invalid choice '{syst_choice}' is not recognized.")
        sys.exit(1)
    
    return syst_map[syst_choice]

def extras():
    print("\nTurn on --plotAll to plot all signal samples?\n")

    print("1. Turn on")
    print("2. Turn off")

    selection_map = {
        '1': '--plotAll',
        '2': '',
    }

    valid_sels = ['1', '2']
            
    sel_choice = input("\nEnter the number corresponding to the selection you desire:\n")

    if sel_choice not in valid_sels:
        print(f"Error: Invalid choice '{sel_choice}' is not recognized.")
        sys.exit(1)

    extra_cmds = input("\nAre there any other plotter_vh arguments you would like to provide? Add them as if you were appending them to the end of the existing command.\n")

    return selection_map[sel_choice], extra_cmds

def extras_Cards():
    print("\nTurn --doAll on to make cards for all signal samples?\n")

    print("1. Turn on")
    print("2. Turn off")

    selection_map = {
        '1': '--doAll',
        '2': '',
    }

    valid_sels = ['1', '2']
            
    sel_choice = input("\nEnter the number corresponding to the selection you desire:\n")

    if sel_choice not in valid_sels:
        print(f"Error: Invalid choice '{sel_choice}' is not recognized.")
        sys.exit(1)   

    print("\nTurn on --ManualMCStats?\n")

    print("1. Turn on")
    print("2. Turn off")

    stats_map = {
        '1': '--ManualMCStats',
        '2': '',
    }

    valid_stats = ['1', '2']
            
    stats_choice = input("\nEnter the number corresponding to the selection you desire:\n")

    if stats_choice not in valid_stats:
        print(f"Error: Invalid choice '{stats_choice}' is not recognized.")
        sys.exit(1) 

    print("\nFloat the background using --floatB?\n")

    print("1. Turn on")
    print("2. Turn off")

    float_map = {
        '1': '--floatB',
        '2': '',
    }

    valid_floats = ['1', '2']
            
    floats_choice = input("\nEnter the number corresponding to the selection you desire:\n")

    if floats_choice not in valid_floats:
        print(f"Error: Invalid choice '{floats_choice}' is not recognized.")
        sys.exit(1) 

    int_choice = int(input("\nTurn on --integrateBins? Any positive non-zero value will be taken as a yes.\n"))

    if int_choice:
        intBins = '--integrateBins ' + str(int_choice)
    else:
        intBins = ''

    extra_cmds = input("\nAre there any other datacardMaker arguments you would like to provide? Add them as if you were appending them to the end of the existing command.\n")

    return selection_map[sel_choice], stats_map[stats_choice], float_map[floats_choice], intBins, extra_cmds

lumi_map = {
    'UL18' : '59.9',
    'UL17' : '41.6',
    'UL16' : '16.4',
    'UL16APV' : '19.9',
    'RunII' : '137.0',
}

def generate_bash_script():
    print("Welcome to the ZH SUEP Script Generator!\n")
    print("Please choose the task you want to include in your script:\n")
    print("1. Saving Histograms")
    print("2. Plotting Histograms")
    print("3. Running Cards")

    # Ask user for their choice
    choice = input("\nEnter the number corresponding to the task you want to perform: ")

    # Initialize script content
    script_content = "#!/bin/bash\n\n"

    # Generate commands based on user choice
    if choice == '1':
        years = whatYear(choice)
        sample = whatSample()
        selection, dimensions = whatPlots(choice)
        
        if dimensions == '2D':
            systs = ''
        else:
            systs = whatSysts()

        out_path = input("\nPlease provide an output path, with the year replaced with 'XXXX'. Ex: /path/to/histos/CRDY_XXXX_Histos\n")
        if 'XXXX' not in out_path:
            print(f"Error: Please format the path to the output histograms with an 'XXXX' substring representing the year.")
            sys.exit(1)

        plotAll, extra_cmds = extras()

        jobs = input("\nWould you like to submit these as jobs? (1 for yes with default values, 2 for yes but let me choose, 3 for no)\n")

        if jobs == '1':
            commands = [f"python plotter_vh.py {sample.replace('XXXX', year)} {selection} -l {lumi_map[year]} {systs} --toSave {out_path.replace('XXXX', year)} {plotAll} --jobName {out_path.replace('XXXX', year)}_Jobs --batchsize 100 --resubmit --queue workday {extra_cmds};" for year in years]
        elif jobs == '2':
            job_info = input("\nPlease specify your job-related custom commands now. Please include at least the --jobName, --batchSize, and --queue, though I will not be error-catching this script if you choose not to.\n")
            commands = [f"python plotter_vh.py {sample.replace('XXXX', year)} {selection} -l {lumi_map[year]} {systs} --toSave {out_path.replace('XXXX', year)} {plotAll} {job_info} --resubmit {extra_cmds};" for year in years]
        elif jobs == '3':
            commands = [f"python plotter_vh.py {sample.replace('XXXX', year)} {selection} -l {lumi_map[year]} {systs} --toSave {out_path.replace('XXXX', year)} {plotAll} {extra_cmds}" for year in years]
        else:
            print(f"Error: Invalid choice '{jobs}' is not recognized.")
            sys.exit(1)

        for i in commands:
            print(i+'\n')

    elif choice == '2':
        years = whatYear(choice)
        sample = whatSample()
        selection, dimensions = whatPlots(choice)

        if dimensions == '2D':
            systs = ''
        else:
            systs = whatSysts()

        histo_path = input("\nPlease provide a path to the existing histograms, replacing the year with 'XXXX'. Ex: /path/to/histos/CRDY_XXXX_Histos\n")
        if 'XXXX' not in histo_path:
            print(f"Error: Please format the path to the histograms with an 'XXXX' substring representing the year.")
            sys.exit(1)

        out_path = input("\nPlease provide an output path, replacing the year with 'XXXX'. Ex: /path/to/plots/CRDY_XXXX_Plots\n")
        if 'XXXX' not in out_path:
            print(f"Error: Please format the path to the output plots with an 'XXXX' substring representing the year.")
            sys.exit(1)

        plotAll, extra_cmds = extras()

        if dimensions == '2D':
            commands = [f"python plotter_vh.py {sample.replace('XXXX', year)} {selection} -l {lumi_map[year]} {systs} --toLoad {histo_path.replace('XXXX', year)} --plotdir {out_path.replace('XXXX', year)} {plotAll} {extra_cmds}" for year in years]
        else:
            commands = [f"python plotter_vh.py {sample.replace('XXXX', year)} {selection} -l {lumi_map[year]} {systs} --toLoad {histo_path.replace('XXXX', year)} --plotdir {out_path.replace('XXXX', year)} {plotAll} --strict-order {extra_cmds}" for year in years]

        for i in commands:
            print(i+'\n')

    elif choice == '3':
        years = whatYear(choice)
        sample = whatSample()
        systs = whatSysts()
        region = whatCards()

        histo_path = input("\nPlease provide a path to the existing root files, replacing the year with 'XXXX'. Ex: /path/to/root_files/CRDY_XXXX_Plots\n")
        if 'XXXX' not in histo_path:
            print(f"Error: Please format the path to the root files with an 'XXXX' substring representing the year.")
            sys.exit(1)

        out_path = input("\nPlease provide an output path for the cards. Ex: /path/to/cards/CRDY_Cards\n")

        doAll, manualMC, floatB, integrateBins, extra_cmds = extras_Cards()
    
        commands = [f"python datacardMaker.py {sample.replace('XXXX', year)} {systs} {out_path}/{year.replace('UL', '20')} --rootFile {histo_path.replace('XXXX', year)}/leadclustertracks --var leadclustertracks --ABCD --year {year.replace('UL', '20')} --region {region} {floatB} {doAll} {manualMC} {integrateBins} {extra_cmds}" for year in years]

        for i in commands:
            print(i+'\n')
    else:
        print(f"Error: '{choice}' is not recognized.")
        sys.exit(1)

# Main program execution
if __name__ == "__main__":
    generate_bash_script()
