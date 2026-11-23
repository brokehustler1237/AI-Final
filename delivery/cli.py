"""Command line interface.

    python3 food_delivery.py                              alert before departure
    python3 food_delivery.py no_alert --truth blocked     courier continues, bridge was blocked
    python3 food_delivery.py --map 10x10 alert --on-the-way
"""

import argparse

from .report import main
from .world import MAPS, use_map


def run(argv=None):
    ap = argparse.ArgumentParser(description="Mini Food Delivery Assistant")
    ap.add_argument("observation", nargs="?", default="alert",
                    choices=["alert", "no_alert", "none"],
                    help="what the traffic service reports about the bridge")
    ap.add_argument("--on-the-way", action="store_true",
                    help="the alert arrives one block before the bridge")
    ap.add_argument("--truth", choices=["clear", "blocked"], default="clear",
                    help="whether the bridge was really blocked (hidden from the assistant)")
    ap.add_argument("--map", choices=list(MAPS), default="6x6",
                    help="which map to use (6x6 is the one from the project brief)")
    args = ap.parse_args(argv)
    use_map(args.map)
    obs = {"alert": "ALERT", "no_alert": "NO ALERT", "none": None}[args.observation]
    main(obs, args.on_the_way, args.truth)


if __name__ == "__main__":
    run()
