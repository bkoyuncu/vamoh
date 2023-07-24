import argparse
import sys


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file path',
        required=False,
        type=str,
        default="./configs/models/chairs.yaml"
    )
    parser.add_argument(
        '--repeat',
        dest='repeat',
        help='Repeat how many random seeds',
        default=1,
        type=int
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Random seed',
        default=0,
        type=int
    )
    parser.add_argument(
        '--sample',
        dest='sample',
        help='only_sampling',
        default=0,
        type=int
    )
    parser.add_argument(
        '--local',
        dest='local',
        help='if local true use cpu for sampling',
        default=0,
        type=int
    )
    parser.add_argument(
        '--mark_done',
        dest='mark_done',
        action='store_true',
        help='mark yaml as yaml_done after a job has finished',
    )
    parser.add_argument(
        '--first_batch',
        dest='first_batch',
        help='inference for first batch',
        default=True,
        type=bool
    )

    parser.add_argument(
        '--save_samples',
        dest='save_samples',
        help='if saving generated samples for metrics',
        default=0,
        type=int
    )

    parser.add_argument(
        "--list_ckpts",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[99],  # default if nothing is provided
    )

    parser.add_argument(
        "--recons",  # name on the CLI - drop the `--` for positional/required parameters
        dest='recons',  # 0 or more values expected => creates a list
        type=int,
        default=1,  # default if nothing is provided
    )
    
    parser.add_argument(
        'opts',
        help='See graphgym/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    return parser.parse_args()
