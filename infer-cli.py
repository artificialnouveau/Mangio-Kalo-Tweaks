import subprocess
import sys
import argparse
from argparse import ArgumentParser

import logging
from logging import getLogger

import json

from pythonosc import dispatcher as Dispatcher, osc_server
from pythonosc import udp_client
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
parser.add_argument("--use-osc", action="store_true", help="Run in OSC mode.")
parser.add_argument("--model", "-m", type=str, required=False, help="Path to model file")
parser.add_argument("--model-index", "-mi", type=str, required=False, help="Path to model index file")
parser.add_argument("--input-file", type=str, required=False, help="Path to input audio file")
parser.add_argument("--output-file", type=str, required=False, help="Path to save processed audio file")
parser.add_argument("--hubert", type=str, default="models/hubert_base.pt", help="Path to Hubert model")
parser.add_argument("--float", action="store_true", help="Use floating point precision")
parser.add_argument("--quality", "-q", type=int, default=1, help="Quality level (default is 1)")
parser.add_argument("--f0-up-key", "-k", type=int, default=0, help="F0 up key value")
parser.add_argument("--f0-method", type=str, default="pm", choices=("pm", "harvest", "crepe", "crepe-tiny"), help="Method for F0 determination")
parser.add_argument("--buffer-size", type=int, default=1000, help="Buffering size in ms")


def set_all_paths(address, args_string, analyze=False):
    global osc_args
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
        
        # For the remaining paths, order is: model_folder, output1, model_folder, output2, ...
        for i in range(1, len(paths)-1, 2):
            model_folder = paths[i]
            # Search for .pth and .index files in the model folder
            for file in os.listdir(model_folder):
                if file.endswith(".pth"):
                    models.append(os.path.join(model_folder, file))
                elif file.endswith(".index"):
                    models_index.append(os.path.join(model_folder, file))

            output_files.append(paths[i + 1])

        # Ensure the input_files list has the same length as models and output_files
        input_files = [input_file] * len(models)

        osc_args["input_files"] = input_files
        osc_args["output_files"] = output_files
        osc_args["models"] = models
        osc_args["models_index"] = models_index

        print("input_files: ", osc_args["input_files"])
        print("output_files: ", osc_args["output_files"])
        print("models: ", osc_args["models"])
        print("models index: ", osc_args["models_index"])

        # Now, send the information to the RVC infer app
        for idx in range(len(models)):
            # Create a dummy args object for the send_to_rvc function
            osc_command = argparse.Namespace()
            osc_command.model = models[idx]
            osc_command.input_file = input_files[idx]
            osc_command.output_file = output_files[idx]
            osc_command.model_index = models_index[idx]
            # Set default values
            osc_command.args_defaults = "0 -2 harvest 160 3 0 1 0.95 0.33"
            send_to_rvc(osc_command)

    except IndexError:
        print("Incorrect sequence of arguments received. Expecting input_path, followed by alternating model_folder and output_path.")


def run_osc_server(args):
    disp = Dispatcher()
    disp.map("/max2py", set_all_paths)  # One OSC address to set all paths

    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 1111), disp)
    print(f"Serving on {server.server_address}")

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
                                    bufsize=1,
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
    try:
        global rvc_process
        print("Entered send_to_rvc function.")
        logger.info(f"Sending command to RVC: {command_str}")
    
        if rvc_process:
            # Construct the command string to send to the Mangio-RVC-Fork v2 CLI App
            command_str = (f"{args.model} {args.input_file} {args.output_file} "
                           f"{args.model_index} {args.args_defaults}\n")
    
            # Send the command to the process
            rvc_process.stdin.write(command_str)
            rvc_process.stdin.flush()
            logger.info(f"Sending command to RVC: {command_str}")
    
            # Optionally: Wait and read the output, if required
            while True:
                line = rvc_process.stdout.readline()
                if "some expected output line" in line:  # Replace with actual expected end line if any
                    break
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()



if __name__ == "__main__":
    print("Starting the RVC process...")
    start_rvc_process()  # Start the RVC process immediately
    print("RVC process started.")

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    # Check the analyze flag and process accordingly
    if args.analyze:
        if not args.input_file and not args.use_osc:
            print("The --input-file option is required for analysis when not using OSC mode.")
            sys.exit(1)
        if args.input_file:
            json_result = analyze_audio(args.input_file)
            print(f"Analysis saved in: {json_result}")

    # Check if OSC mode is active
    if args.use_osc:
        print("Running in OSC mode.")
        run_osc_server(args)
    else:
        print("Running in non-OSC mode.")
        if not args.model or not args.input_file or not args.output_file:
            print("Required arguments missing for non-OSC mode.")
            sys.exit(1)
        send_to_rvc(args)
