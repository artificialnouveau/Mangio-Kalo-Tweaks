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
from rvc_eval.vc_infer_pipeline import VC
from rvc_eval.model import load_hubert, load_net_g

sys.path.append(os.path.dirname(__file__))
from speech_analysis import analyze_audio

sys.path.append(os.path.join(os.path.dirname(__file__), "..\\rvc\\"))

# Assuming global logger and parser configurations
logger = getLogger(__name__)

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
parser.add_argument("--analyze", action="store_true", help="Analyze the input audio file")


def set_all_paths(address, args_string, analyze=True):  # 'analyze' parameter is retained
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
        
        # For the remaining paths, order is: model1, output1, model2, output2, ...
        for i in range(1, len(paths)-1, 2):
            models.append(paths[i])
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

    except IndexError:
        print("Incorrect sequence of arguments received. Expecting input_path, followed by alternating model_path and output_path.")


def run_osc_server(args):
    disp = Dispatcher()
    disp.map("/max2py", set_all_paths)  # One OSC address to set all paths

    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 1111), disp)
    print(f"Serving on {server.server_address}")

    def handle_requests():
        while True:
            server.handle_request()
            
            # Send to the Mangio-RVC subprocess for each model, input, and output path
            for model_path, model_index_path, input_path, output_path in zip(osc_args["models"], osc_args["models_index"], osc_args["input_files"], osc_args["output_files"]):
                args.model = model_path.replace('"', '')
                args.model_index = model_index_path.replace('"', '')
                args.input_file = input_path.replace('"', '')
                args.output_file = output_path.replace('"', '')
                send_to_rvc(args)

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
            line = rvc_process.stdout.readline()
            if "You are currently in 'HOME':" in line:
                break

        # Send 'go infer' to enter infer mode
        rvc_process.stdin.write("go infer\n")
        rvc_process.stdin.flush()

        # Wait for "INFER:" prompt
        while True:
            line = rvc_process.stdout.readline()
            if "INFER:" in line:
                break

def send_to_rvc(args):
    global rvc_process
    if rvc_process:
        # Construct the command string to send to the Mangio-RVC-Fork v2 CLI App
        command_str = (f"{args.model} {args.model_index} {args.input_file} {args.output_file}"
                       f"{args.feature_index_file_path} {args.speaker_id} {args.transposition} "
                       f"{args.f0_method} {args.crepe_hop_length} {args.harvest_median_filter_radius} "
                       f"{args.post_resample_rate} {args.mix_volume_envelope} {args.feature_index_ratio} "
                       f"{args.vc_protection}\n")

        # Send the command to the process
        rvc_process.stdin.write(command_str)
        rvc_process.stdin.flush()

        # Optionally: Wait and read the output, if required
        while True:
            line = rvc_process.stdout.readline()
            if "some expected output line" in line:  # Replace with actual expected end line if any
                break

def main(args):
    try:
        logger.info("Starting main function...")
        
        # Assuming you have some function called process_file
        # that does something with the input file
        if args.input_file:
            result = process_file(args.input_file)
            logger.info(f"Processed file with result: {result}")
        
        # Send OSC command only if --use-osc argument is provided
        if args.use_osc:
            # Create a client to send OSC messages
            sender = udp_client.SimpleUDPClient("127.0.0.1", 6666) # Remote: 192.168.2.110
            
            _out = args.output_file
            _mod = args.model
            _modidx = args.model_index
            
            message = 'output file: ' + _out + ' with ' + _mod + ' is done.'
            sender.send_message("/py2max/gen_done", message)

            # Construct and send the OSC command for Mangio-RVC-Fork v2
            infer_command = f"{_mod} {args.input_file} {_out} logs/{_modidx} 0 0 harvest 160 3 0 1 0.95 0.33 False False False"
            sender.send_message("/py2max/infer", infer_command)

        # If not using OSC, use subprocess to run the Mangio-RVC-Fork v2 CLI command
        else:
            import subprocess
            
            infer_command = [
                "python", "infer-web.py", "--pycmd", "python", "--is_cli",
                _mod, args.input_file, _out,
                f"logs/{_modidx}",
                "0", "0", "harvest", "160", "3", "0", "1", "0.95", "0.33", "False", "False", "False"
            ]
            subprocess.run(infer_command)
    
    except Exception as e:
        logger.error(f"Error encountered: {e}")
        sys.exit(1)

    logger.info("Main function completed successfully!")


if __name__ == "__main__":
    start_rvc_process()

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.analyze:
        if not args.input_file and not args.use_osc:
            print("The --input-file option is required for analysis when not using OSC mode.")
            sys.exit(1)
        if args.input_file:
            json_result = analyze_audio(args.input_file)
            print(f"Analysis saved in: {json_result}")

    if args.use_osc:
        run_osc_server(args)
    else:
        if not args.model or not args.input_file or not args.output_file:
            print("When not using OSC mode, -m/--model, --input-file, and --output-file are required.")
            sys.exit(1)
        send_to_rvc(args)

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.analyze:
        if not args.input_file and not args.use_osc:
            print("The --input-file option is required for analysis when not using OSC mode.")
            sys.exit(1)
        if args.input_file:
            json_result = analyze_audio(args.input_file)
            print(f"Analysis saved in: {json_result}")

    if not args.use_osc:
        if not args.model or not args.input_file or not args.output_file:
            print("When not using OSC mode, -m/--model, --input-file, and --output-file are required.")
            sys.exit(1)
        send_to_rvc(args)
    else:
        if not args.model or not args.input_file or not args.output_file:
            print("When not using OSC mode, -m/--model, --input-file, and --output-file are required.")
            sys.exit(1)
        main(args)
