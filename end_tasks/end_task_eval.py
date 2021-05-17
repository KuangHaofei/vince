import torch.multiprocessing as multiprocessing
from dg_util.python_utils import tensorboard_logger
import arg_parser
import os


def main():
    args = arg_parser.parse_args()
    val_logger = tensorboard_logger.Logger(os.path.join(args.tensorboard_dir))
    solver = args.solver(args, None, val_logger)
    solver.run_eval()


if __name__ == "__main__":
    cxt = multiprocessing.get_context()
    print(cxt.get_start_method())
    main()
