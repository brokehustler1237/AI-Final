import os
import sys

# make the delivery package importable when pytest runs from any folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
