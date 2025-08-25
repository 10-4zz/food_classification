import argparse


def get_options():
    parser = argparse.ArgumentParser(description="Food Classification Options")

    # Add arguments for various options
    parser.add_argument('--cfg', type=str, default=None, help='Path to config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Override config options')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--image_size', type=int, default=224, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, default='ETHZFOOD101', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='af_net', help='Name of the model')
    parser.add_argument('--cache_mode', type=str, default='no_cache', help='Cache mode for dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use checkpoint during training')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--tag', type=str, default='', help='Tag for the experiment')
    parser.add_argument('--eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--throughput', action='store_true', help='Enable throughput mode')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=True,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9996,
                        help='decay factor for model weights moving average (default: 0.9996)')

    return parser
