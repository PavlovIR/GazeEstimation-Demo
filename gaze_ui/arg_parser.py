import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time/image inference for the iTracker model."
    )
    parser.add_argument(
        "--checkpoint", default="checkpoint.pth.tar", help="Path to checkpoint.pth.tar"
    )
    parser.add_argument(
        "--image",
        help="Optional path to an image. If omitted, runs on webcam in real time.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--device", default="auto", help="cpu, cuda, or auto (default)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display frames when running on webcam.",
    )
    parser.add_argument(
        "--accuracy-ui",
        action="store_true",
        help="Open a pygame window with a target dot to estimate accuracy.",
    )
    parser.add_argument(
        "--accuracy-grid-ui",
        action="store_true",
        help="Open a pygame window with a visual-angle grid to estimate accuracy in degrees.",
    )
    parser.add_argument(
        "--distance-cm",
        type=float,
        default=60.0,
        help="Viewer distance from screen in cm (used with --accuracy-grid-ui).",
    )
    parser.add_argument(
        "--grid-spacing-deg",
        type=float,
        default=2.0,
        help="Grid spacing in degrees (used with --accuracy-grid-ui).",
    )
    parser.add_argument(
        "--threshold-deg",
        type=float,
        default=1.0,
        help="Accuracy threshold in degrees (used with --accuracy-grid-ui).",
    )
    parser.add_argument(
        "--grid-hold-seconds",
        type=float,
        default=2.0,
        help="Seconds to hold each target in the grid UI.",
    )
    parser.add_argument(
        "--demo-ui",
        action="store_true",
        help="Open a pygame window with blocks for demo of gaze estimations.",
    )
    parser.add_argument(
        "--screen-config",
        help="Path to .toml or .json file with screen parameters (used with pygame UIs).",
    )
    parser.add_argument(
        "--log",
        nargs="?",
        const="accuracy_runs",
        help=(
            "Write accuracy run logs to this directory "
            "(default: accuracy_runs). Requires --accuracy-ui or --accuracy-grid-ui."
        ),
    )
    return parser.parse_args()
