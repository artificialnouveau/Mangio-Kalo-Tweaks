import subprocess
import sys
import shutil
import argparse
from argparse import ArgumentParser

import logging
from logging import getLogger

import json

from pythonosc.dispatcher import Dispatcher
from pythonosc import udp_client, osc_server
import traceback

import time
import os
from scipy.io.wavfile import read, write
from scipy.signal import resample_poly

from threading import Thread

import soundfile as sf
import numpy as np
import torch
from tqdm import tqdm

# Assuming global logger and parser configurations
logger = getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def print_handler(address, *args):
    print(f"Received message from {address}: {args}")

parser = ArgumentParser(description="Voice Conversion CLI Application")

# Existing arguments:
parser.add_argument("--model", "-m", type=str, required=False, help="Path to model file")
parser.add_argument("--model-index", "-mi", type=str, required=False, help="Path to model index file")
parser.add_argument("--input-file", type=str, required=False, help="Path to input audio file")
parser.add_argument("--output-file", type=str, required=False, help="Path to save processed audio file")
parser.add_argument("--use-osc", action="store_true", help="Run in OSC mode.")
parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")


def set_all_paths(address, args_string, analyze=False):
    logger.info(f"Received OSC command at address {address} with arguments: {args_string}")
    global osc_args
    if 'osc_args' not in globals():
        osc_args = {}
    if args_string.startswith("'") and args_string.endswith("'"):
        args_string = args_string[1:-1]
    if 'Macintosh HD:' in args_string:
        args_string = args_string.replace('Macintosh HD:', '')
        
    paths = args_string.split(", ")

    input_files = []
    output_files = []
    models = []
    models_index = []

    try:
        # The first path is always input
        input_file = paths[0]
        for i in range(1, len(paths) - 1, 2):
            model_folder = paths[i]
            # Search for .pth and .index files in the model folder
            for file in os.listdir(model_folder):
                if file.endswith(".pth"):
                    models.append(os.path.join(model_folder, file))
                elif file.endswith(".index"):
                    models_index.append(os.path.join(model_folder, file))
    
            output_files.append(paths[i + 1])
    
        input_files = [input_file] * len(models)
    
        osc_args = {
            "input_files": input_files,
            "output_files": output_files,
            "models": models,
            "models_index": models_index,
        }

        for idx in range(len(models)):
            model_name = os.path.basename(models[idx])
            model_path_in_weights = os.path.join("./weights", model_name)
            
            # Check if the model exists in the ./weights folder
            if not os.path.exists(model_path_in_weights):
                shutil.copy(models[idx], model_path_in_weights)

            log_dir = os.path.join("logs", model_name.split(".")[0])  # get the model name without extension
            original_index_file_name = os.path.basename(models_index[idx])
            destination_index_file_path = os.path.join(log_dir, original_index_file_name)
    
            # Check if the directory exists, if not, create it
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
    
            # Check if the destination index file exists, if not, copy the original index file to it
            if not os.path.exists(destination_index_file_path):
                shutil.copy(models_index[idx], destination_index_file_path)
    
            # Create a dummy args object for the send_to_rvc function
            osc_command = argparse.Namespace()
            osc_command.model = model_name  # Only the model name as input for the model argument
            osc_command.input_file = input_files[idx] # "input_audio/"+os.path.basename(input_files[idx])
            osc_command.output_file = os.path.basename(output_files[idx])
            osc_command.model_index = destination_index_file_path  # Use the copied index path
            # Set default values
            osc_command.args_defaults = "0 -2 harvest 128 3 0 1 0.95 0.33 False False" # 0.33
            send_to_rvc(osc_command)

    except IndexError:
        print("Incorrect sequence of arguments received. Expecting input_path, followed by alternating model_folder and output_path.")


def run_osc_server(args):
    disp = Dispatcher()
    disp.map("/max2py", set_all_paths)  # One OSC address to set all paths

    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 1111), disp)
    print(f"Serving on {server.server_address}")
    logger.info(f"OSC server started and listening on {server.server_address}")

    def handle_requests():
        while True:
            server.handle_request()

    # Run the server in a separate thread
    thread = Thread(target=handle_requests)
    thread.start()


def resample_audio(audio, original_sr, target_sr):
    from math import gcd

    # Calculate up and down sampling ratios
    g = gcd(original_sr, target_sr)
    up = target_sr // g
    down = original_sr // g

    return resample_poly(audio, up, down)


# Global variable to manage the subprocess 
rvc_process = None

def start_rvc_process():
    global rvc_process
    if rvc_process is None:
        rvc_process = subprocess.Popen(['python', 'infer-web.py', '--pycmd', 'python', '--is_cli'],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       text=True,
                                       bufsize=1,  # <-- Enable line buffering
                                       universal_newlines=True)


        # Wait until the welcome message appears
        while True:
            for line in iter(rvc_process.stdout.readline, ''):
                print(line, end='')
                if "You are currently in 'HOME':" in line:
                    rvc_process.stdin.write("go infer\n")
                    rvc_process.stdin.flush()
                elif "INFER:" in line:
                    print("Awaiting OSC commands...")
                    break

        # Wait for "INFER:" prompt
        while True:
            line = rvc_process.stdout.readline()
            if "INFER:" in line:
                print("Awaiting OSC commands...")
                break

def send_to_rvc(args):
    logger.info("Starting send_to_rvc function...")
    try:
        global rvc_process
        print("Entered send_to_rvc function.")
    
        if rvc_process:
            # Wait for the specific text to appear in the RVC process output
            print("Waiting for RVC to be ready...")
            while True:
                line = rvc_process.stdout.readline()
                print(line, end='')  # Optional: print the RVC output
                if "INFER:" in line:
                    print("RVC is ready. Sending OSC command...")
                    break
    
            # Construct the command string to send to the Mangio-RVC-Fork v2 CLI App
            command_str = (f"{args.model} {args.input_file} {args.output_file} "
                           f"{args.model_index} {args.args_defaults}\n")
    
            # Send the command to the process
            logger.info(f"Sending command to RVC: {command_str}")
            rvc_process.stdin.write(command_str)
            rvc_process.stdin.flush()
            logger.info("Command sent.")
    
            # Optionally: Wait and read the output, if required
            while True:
                line = rvc_process.stdout.readline()
                if "some expected output line" in line:  # Replace with actual expected end line if any
                    break
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
    logger.info("Finished send_to_rvc function.")


def start_rvc_process_threaded():
    """Runs the RVC process in a separate thread."""
    global rvc_process_thread
    rvc_process_thread = Thread(target=start_rvc_process)
    rvc_process_thread.start()

if __name__ == "__main__":
    print("Starting the RVC process...")
    start_rvc_process_threaded()
    print("RVC process started.")

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level))
    
    if args.use_osc:
        print("Running in OSC mode.")
        run_osc_server(args)
    else:
        print("Running in non-OSC mode.")
        if not args.model or not args.input_file or not args.output_file:
            print("Required arguments missing for non-OSC mode.")
            sys.exit(1)
        send_to_rvc(args)
