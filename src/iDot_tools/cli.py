#!/usr/bin/env python

import argparse
from .app import iDotScheduler
from . import __version__


def main():
    parser = argparse.ArgumentParser(description="iDot Scheduler GUI")
    parser.add_argument(
        "--version", "-v", action="version", version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args()
    app = iDotScheduler()
    app.launch(inbrowser=True, share=False)


if __name__ == "__main__":
    main()
