#!/usr/bin/env python3
"""
This module is used to empty the log file.
"""

from pathlib import Path

path_list = Path("app").glob("*.log")
for path in path_list:
    print(f"Empty logfile in {path}")
    with open(path, "w"):
        pass

print("Done Empty Log Service")
