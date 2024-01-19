import os                                                                       
import sys                                                                      
import argparse
import itertools as it

import subprocess
from joblib import Parallel, delayed

if __name__ == '__main__':
    # Setting up the parser
    parser = argparse.ArgumentParser(
        description="An analyst for quick ML applications.", add_help=True)

    # Configuring the experiments ----------------------------------------------
    parser.add_argument('-experiment',action='store',dest='EXPERIMENT', type=str,              
                        default="evaluate_model",help=".py experiment file")
    parser.add_argument('-dataset',action='store',dest='DATASET',type=str,              
                        default=(
                            'd_airfoil,'
                            'd_concrete,'
                            'd_enc,'
                            'd_enh,'
                            'd_housing,'
                            # 'd_tower,'
                            # 'd_uball5d,'
                            'd_yacht'
                            ))
    parser.add_argument('-repeats',action='store',dest='REPEATS', type=int,              
                        default=1)
    parser.add_argument('-repeat_number',action='store',dest='REPEAT_N', 
                        type=int, default=-1)
    parser.add_argument('-n_trials',action='store',dest='N_TRIALS',default=1,  
                        type=int, help='Number of trials to run')         
    parser.add_argument('-models',action='store',dest='MODELS', type=str,              
                        default=('BrushClassifier,'
                                'DecisionTree,'
                                'BrushClassifier,'
                                'RandomForest,'
                                'GaussianNaiveBayes,'
                                'LogisticRegression_L1,'
                                'LogisticRegression_L2'
                                ))
    parser.add_argument('--long',action='store_true',dest='LONG', default=False)
    parser.add_argument('-seeds',action='store',type=str,dest='SEEDS',          
            default='14724,24284,31658,6933,1318,16695,27690,8233,24481,6832,'  
                    '13352,4866,12669,12092,15860,19863,6654,10197,29756,14289,'        
                    '4719,12498,29198,10132,28699,32400,18313,26311,9540,20300,'        
                    '6126,5740,20404,9675,22727,25349,9296,22571,2917,21353,'           
                    '871,21924,30132,10102,29759,8653,18998,7376,9271,9292')
    parser.add_argument('-results',action='store',dest='RDIR',                  
                        default='results',type=str,help='Results directory')  
    parser.add_argument('-data-dir',action='store',dest='DDIR',                  
                        default='./',type=str,help='Input data directory')     
    
    # Running locally or submitting to a cluster -------------------------------
    parser.add_argument('--local', action='store_true', dest='LOCAL', 
                        default=False, help='Run locally instead of on HPC')
    parser.add_argument('--dryrun', action='store_true', default=False, 
                        help='Just print fn calls')  
    
    parser.add_argument('--slurm',action='store_true',dest='SLURM',default=False, 
                        help='Run on a SLURM scheduler as opposed to on LPC')
    parser.add_argument('-A', action='store', dest='A', default='plgsrbench', 
                        help='SLURM account')
    parser.add_argument('-n_jobs',action='store',dest='N_JOBS',default=1,          
                        type=int, help='Number of parallel threads to use per job')              
    parser.add_argument('-m',action='store',dest='M',default=8000,type=int,        
                        help='LSF memory request and limit (MB)')
    parser.add_argument('-time', action='store', dest='time', 
                        default='12:00:00', type=str, help='time in HR:MN:SS')
    parser.add_argument('-max_jobs', action='store', default=3500, 
                        type=int, help='Maximum size of job array (for slurm)')
    parser.add_argument('-q', action='store', dest='QUEUE',
                        default=None, type=str, help='queue name')
    
    # Parsing and extracting experiment settings -------------------------------
    args = parser.parse_args()
    
    n_trials = len(args.SEEDS) if args.N_TRIALS < 1 else args.N_TRIALS
    
    print('EXPERIMENT: ', args.EXPERIMENT)
    print('n_trials: ', n_trials)
    print('n_jobs: ', args.N_JOBS)

    seeds = args.SEEDS.split(',')[:n_trials]                          
    
    print('using these seeds:',seeds)
    print('for # of repeats:',args.REPEATS)

    models = args.MODELS.split(',')
    print('for models:',models)
    
    if args.REPEAT_N != -1:
        repeats = [args.REPEAT_N]
        if args.REPEATS != 1:
            raise ValueError('if -repeat_number is specified, '
                    '-repeats must be 1')
    else:
        repeats = range(args.REPEATS)

    datasets = args.DATASET.split(',')
    print('and these datasets:',datasets)                                        

    if not args.LOCAL and args.QUEUE is None:
        if args.SLURM: 
            print('setting queue to bch-compute')
            args.QUEUE = 'bch-compute'
        else:
            args.QUEUE = 'mgh'
                  
    input("Press Enter to continue...")

    # Generating the jobs ------------------------------------------------------
    all_commands = []
    job_info = [] 
    for repeat, dataset, ml in it.product(repeats, datasets, models):
        filepath = '/'.join([args.RDIR,dataset,ml]) + '/'

        if not os.path.exists(filepath):
            print('WARNING: creating path',filepath)
            os.makedirs(filepath)
        
        for seed in seeds:
            random_state = seed
            all_commands.append( #Defining how we'll call the script
                'python {CURRDIR}/{EXPERIMENT}.py '
                ' -ml {ML}'
                ' -dataset {DATASET}'
                ' -seed {RS}'
                ' -rdir {RDIR}'
                ' -repeat {REPEAT}'
                ' -datadir {DDIR}'.format(
                    CURRDIR=os.path.abspath(os.getcwd()),
                    EXPERIMENT=args.EXPERIMENT,
                    ML=ml,
                    DATASET=dataset,
                    RS=random_state,
                    RDIR=filepath,
                    REPEAT=repeat,
                    DDIR=args.DDIR
                )
            )  
                        
            job_info.append({
                'ml':ml,
                'dataset':dataset,
                'repeat':repeat,
                'results_path':filepath,
                'seed':random_state
                })

    # Submiting jobs -----------------------------------------------------------
    print('\n'.join(all_commands))
    if args.dryrun:
        exit()
    if args.LOCAL:   # run locally
        Parallel(n_jobs=args.N_JOBS)(
            delayed(os.system)(run_cmd) for run_cmd in all_commands)
            # delayed(print)(run_cmd) for run_cmd in all_commands)
    else:
        if args.SLURM:
            # write a jobarray file to read commans from
            jobarrayfile = 'jobfiles/joblist.txt'
            for i, run_cmd in enumerate(all_commands):
                mode='w' if i == 0 else 'a'
                with open(jobarrayfile,mode) as f:
                    f.write(f'{run_cmd}\n')

            # job_name = '_'.join([f'{job_info[i][x]}' for x in
            #                      ['ml','dataset','seed']])
            job_name='car-t'
            job_file = f'jobfiles/{job_name}'
            out_file = job_info[i]['results_path'] + job_name + '_%J.out'
            # error_file = out_file[:-4] + '.err'

            mem = args.M
            if len(all_commands)>args.max_jobs:
                joblimit = f'%{args.max_jobs}'
            else:
                joblimit=''

            batch_script = (
f"""#!/usr/bin/bash 
#SBATCH --output=jobfiles/cart_%A_%a.txt 
#SBATCH --job-name={job_name} 
#SBATCH --partition={args.QUEUE} 
#SBATCH --cpus-per-task={args.N_JOBS} 
#SBATCH --time={args.time}
#SBATCH --mem-per-cpu={mem} 
#SBATCH --array=0-{len(all_commands)}

row="$SLURM_ARRAY_TASK_ID"
declare -i row
row+=1
echo "head -$row {jobarrayfile} | tail -n +$row"
cmd=$(head -$row {jobarrayfile} | tail -n +$row)
echo $cmd
$cmd
"""
            )

            with open(job_file,'w') as f:
                f.write(batch_script)

            print(job_file,':')
            print(batch_script)
            sbatch_response = subprocess.check_output(
                [f'sbatch {job_file}'], shell=True).decode()
            print(sbatch_response)
        else: #bsub
            for i, run_cmd in enumerate(all_commands):
                job_name = '_'.join([f'{job_info[i][x]}' for x in
                                     ['ml','dataset','seed']])
                job_file = f'jobfiles/{job_name}'
                out_file = job_info[i]['results_path'] + job_name + '_%J.out'
                # error_file = out_file[:-4] + '.err'

                mem = args.M
                bsub_cmd = ('bsub -o {OUT_FILE} -n {N_CORES} -J {JOB_NAME} '
                            '-q {QUEUE} -R "span[hosts=1] rusage[mem={M}]" '
                            '-M {M} ').format(
                    OUT_FILE=out_file,
                    JOB_NAME=job_name,
                    QUEUE=args.QUEUE,
                    N_CORES=args.N_JOBS,
                    M=mem
                )

                bsub_cmd += '"' + run_cmd + '"'
                print(bsub_cmd)
                os.system(bsub_cmd)     # submit jobs

    print('### Finished submitting',len(all_commands),'jobs. ###')
