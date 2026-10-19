import os
# One BLAS/OpenMP thread per process; parallelism comes from --num_workers.
# Must be set before numpy is imported, otherwise every worker spawns one thread per core.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import argparse
import json
import random

from diffusion_planner.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):

    scenario_types = None

    scenario_tokens                      # List of scenario tokens to include
    log_names = log_names                # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None         # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True              # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = False          # Whether to remove scenarios where the mission goal is invalid
    shuffle                              # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', default='/data/nuplan-v1.1/trainval', type=str, help='path to raw data')
    parser.add_argument('--map_path', default='/data/nuplan-v1.1/maps', type=str, help='path to map data')

    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=10, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='shuffle scenarios')

    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)

    parser.add_argument('--num_workers', type=int, help='number of parallel processes', default=len(os.sched_getaffinity(0)))
    parser.add_argument('--num_shards', type=int, help='split scenarios into this many shards (e.g. one per node)', default=1)
    parser.add_argument('--shard_id', type=int, help='index of the shard to process on this node, in [0, num_shards)', default=0)
    parser.add_argument('--max_tasks_per_child', type=int, help='restart each worker after this many scenarios to release cached memory', default=500)
    parser.add_argument('--seed', type=int, help='random seed for scenario sampling; must match across shards', default=0)
    args = parser.parse_args()
    assert 0 <= args.shard_id < args.num_shards, f"shard_id must be in [0, {args.num_shards})"

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    sensor_root = None
    db_files = None

    # Only preprocess the training data
    with open('./nuplan_train.json', "r", encoding="utf-8") as file:
        log_names = json.load(file)

    map_version = "nuplan-maps-v1.0"    
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    # Fixed seed so every shard samples and shuffles the same scenario list
    random.seed(args.seed)
    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=args.num_workers)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # Split into contiguous shards of the log-sorted list, so each log stays (mostly) on one node
    scenarios = sorted(scenarios, key=lambda s: (s.log_name, s.token))
    shard_size = -(-len(scenarios) // args.num_shards)
    shard_scenarios = scenarios[args.shard_id * shard_size:(args.shard_id + 1) * shard_size]
    if args.num_shards > 1:
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(shard_scenarios)} scenarios")

    # process data
    del worker, builder, scenario_filter
    processor = DataProcessor(args)
    processor.work(shard_scenarios, num_workers=args.num_workers, max_tasks_per_child=args.max_tasks_per_child)

    # Only write the file list once every shard has finished, so it matches a single-node run
    existing_files = set(os.listdir(args.save_path))
    num_missing = sum(f"{s._map_name}_{s.token}.npz" not in existing_files for s in scenarios)
    if num_missing > 0:
        print(f"{num_missing} scenarios from other shards are not finished yet; "
              f"the last shard to finish writes diffusion_planner_training.json")
    else:
        npz_files = sorted(f for f in existing_files if f.endswith('.npz'))

        # Save the list to a JSON file
        with open('./diffusion_planner_training.json', 'w') as json_file:
            json.dump(npz_files, json_file, indent=4)

        print(f"Saved {len(npz_files)} .npz file names")
