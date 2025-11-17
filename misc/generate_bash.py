from pathlib import Path

top_dir = Path(__file__).parent.parent
pkg_dir = Path(__file__).parent.parent.parent.parent

bash_template_0 = \
"""\
repo="{top_dir}"

# Create logs directory and set up logging
mkdir -p $repo/logs
exec > >(tee -a $repo/logs/{log_filename}) 2>&1

echo "Running Python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/alexey/.virtualenvs/ode-lstms/bin/activate

"""

bash_template_1 = \
"""\
export CUDA_VISIBLE_DEVICES={cuda_device} && python -u $repo/run_opt.py "{config_path}"

"""

bash_template_2 = \
"""\
echo "Finished running Python"

"""

# Parameters
n_parallel = 2
models = ["ode-lstm", "lstm"]
datasets = ["ocean_das"]
pair_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
validation = 0

# Create and clean up bash repo
bash_dir = top_dir / 'bash'
bash_dir.mkdir(exist_ok=True)
for file in bash_dir.glob('*.sh'):
    file.unlink()

device_counter = 0
skipped_counter = 0
devices = ["cuda:1"]
total_scripts = len(devices) * n_parallel

# Initialize bash scripts for each device and parallel index
bash_scripts = {}
for device in devices:
    for parallel_idx in range(n_parallel):
        script_key = f"{device}_{parallel_idx}"
        device_num = device.split(':')[1]
        log_filename = f"run_cuda_{device_num}_{parallel_idx}.log"
        bash_scripts[script_key] = bash_template_0.format(log_filename=log_filename, top_dir=top_dir)

for dataset in datasets:
    for model in models:
        for pair_id in pair_ids:
            # Check if results already exist
            results_dir = pkg_dir / "results" / "tune_results" / model / dataset / f"pair_id_{pair_id}"
            if list(results_dir.glob(f"**/optimal_params_{dataset}_{pair_id}.yaml")):
                skipped_counter += 1
                continue
            
            config_path = top_dir / 'tuning_config' / f'config_{model}_{dataset}_{pair_id}.yaml'

            # Determine which device and parallel script to use based on counter
            device_idx = device_counter % len(devices)
            parallel_idx = (device_counter // len(devices)) % n_parallel
            current_device = devices[device_idx]
            device_num = current_device.split(':')[1]
            script_key = f"{current_device}_{parallel_idx}"
            
            cmd = bash_template_1.format(
                config_path=config_path,
                cuda_device=device_num,
            )

            # Add the command to the appropriate bash script
            bash_scripts[script_key] += cmd

            print(f"Adding {model}/{dataset}/pair_id_{pair_id} to {script_key}")

            device_counter += 1

# Add the closing template to each script and write to files
for script_key, script_content in bash_scripts.items():
    script_content += bash_template_2
    
    # Parse device and parallel index from script_key
    device, parallel_idx = script_key.rsplit('_', 1)
    device_num = device.split(':')[1]  # Extract number from "cuda:X"
    filename = f"run_cuda_{device_num}_{parallel_idx}.sh"
    filepath = bash_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    filepath.chmod(0o755)
    
    print(f"Generated bash script: {filepath}")

total_possible_jobs = len(datasets) * len(models) * len(pair_ids)
jobs_added = device_counter
print(f"\nTotal possible jobs: {total_possible_jobs}")
print(f"Jobs skipped (already completed): {skipped_counter}")
print(f"Jobs added to scripts: {jobs_added}")
print(f"Total scripts generated: {total_scripts}")
if jobs_added > 0:
    print(f"Jobs per script: ~{jobs_added // total_scripts} (with remainder distributed)")
else:
    print(f"Jobs per script: 0 (all jobs already completed)")
