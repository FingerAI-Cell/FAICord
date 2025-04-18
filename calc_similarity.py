from src import WeSPEAKEMB, SBEMB
from dotenv import load_dotenv
import numpy as np
import argparse
import json
import os

def main(args):
    sbemb = SBEMB()
    wsemb = WeSPEAKEMB()
    

if __init__ == '__main__':
    argparser = argparse.ArgumentParser() 
    argparser.add_arugment('--file_name', type=str, required=True)
    cli_args = argparser.parse_args()
    main(cli_args)