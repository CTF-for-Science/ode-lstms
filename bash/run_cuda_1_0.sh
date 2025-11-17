repo="/home/alexey/Git/ctf4science/models/ode-lstms"

# Create logs directory and set up logging
mkdir -p $repo/logs
exec > >(tee -a $repo/logs/run_cuda_1_0.log) 2>&1

echo "Running Python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/alexey/.virtualenvs/ode-lstms/bin/activate

export CUDA_VISIBLE_DEVICES=1 && python -u $repo/run_opt.py "/home/alexey/Git/ctf4science/models/ode-lstms/tuning_config/config_ode-lstm_ocean_das_1.yaml"

export CUDA_VISIBLE_DEVICES=1 && python -u $repo/run_opt.py "/home/alexey/Git/ctf4science/models/ode-lstms/tuning_config/config_lstm_ocean_das_1.yaml"

export CUDA_VISIBLE_DEVICES=1 && python -u $repo/run_opt.py "/home/alexey/Git/ctf4science/models/ode-lstms/tuning_config/config_lstm_ocean_das_8.yaml"

echo "Finished running Python"

