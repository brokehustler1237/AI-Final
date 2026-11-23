"""Mini Food Delivery Assistant: run every stage and print it to the console.

    python3 food_delivery.py                              alert before departure
    python3 food_delivery.py no_alert                     no alert from the traffic service
    python3 food_delivery.py none                         no report at all
    python3 food_delivery.py alert --on-the-way           alert one block before the bridge
    python3 food_delivery.py no_alert --truth blocked     courier continues, bridge was blocked
    python3 food_delivery.py --map 10x10                  bigger map with a river

The code lives in the delivery/ package; this file is only the entry point.
"""

from delivery.cli import run

if __name__ == "__main__":
    run()
