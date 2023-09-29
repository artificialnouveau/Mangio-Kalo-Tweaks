import subprocess
import sys
import argparse
from argparse import ArgumentParser

import logging
import json
from pythonosc import udp_client

# Assuming global logger and parser configurations
logger = logging.getLogger(__name__)

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

def run_mangio_rvc(args):
    process = subprocess.Popen(['python', 'infer-web.py', '--pycmd', 'python', '--is_cli'],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               text=True,
                               bufsize=1,
                               universal_newlines=True)
    
    # Wait until we get to the "INFER:" prompt
    while True:
        line = process.stdout.readline()
        if "INFER:" in line:
            break

    # Now, construct the command string to send to the Mangio-RVC-Fork v2 CLI App
    command_str = (f"{args.model} {args.model_index} {args.input_file} {args.output_file}"
                   f"{args.feature_index_file_path} {args.speaker_id} {args.transposition} "
                   f"{args.f0_method} {args.crepe_hop_length} {args.harvest_median_filter_radius} "
                   f"{args.post_resample_rate} {args.mix_volume_envelope} {args.feature_index_ratio} "
                   f"{args.vc_protection}\n")

    # Send the command to the process
    process.stdin.write(command_str)
    process.stdin.flush()

    # Optionally: Wait and read the output, if required
    while True:
        line = process.stdout.readline()
        if "some expected output line" in line:  # Replace with actual expected end line if any
            break

    process.terminate()


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
        main(args)
