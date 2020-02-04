"""Module containing the path to the data, the logs and the model weights
"""
import os

BSD500_DATA_DIR = os.environ.get('BSD500_DATA_DIR', './')
BSD68_DATA_DIR = os.environ.get('BSD68_DATA_DIR', './')
DIV2K_DATA_DIR = os.environ.get('DIV2K_DATA_DIR', './')
LOGS_DIR = os.environ.get('LOGS_DIR', './')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', './')
