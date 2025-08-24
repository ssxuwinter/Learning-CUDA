
srun --gres=gpu:nvidia:2 --cpus-per-task=16 --mem=256G make VERBOSE=true > "log/log_$(date '+%Y-%m-%d_%H-%M-%S').txt"
