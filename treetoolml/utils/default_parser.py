import argparse


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.
    Args:
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--mp", default=0, type=int, metavar="int", help="path to config file"
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--device", type=str, default='gpu', help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--gpu_number", type=int, default=0, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--amp", type=int, default=0, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--resume", type=int, default=0, help="resume from checkpoint"
    )
    return parser
