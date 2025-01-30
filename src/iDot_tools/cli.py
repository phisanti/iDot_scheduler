#!/usr/bin/env python

import argparse
from .app import iDotScheduler

def main():
    parser = argparse.ArgumentParser(description='iDot Scheduler GUI')
    args = parser.parse_args()
    app = iDotScheduler()
    app.launch(
        inbrowser=True,  
        share=False      
    )

if __name__ == "__main__":
    main()