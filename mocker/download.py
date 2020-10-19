#!/usr/bin/env python3
import subprocess
import click
from pathlib import Path
import shutil
import logging
import sys
import time

def prompt(text, choices):
    """Will repeat prompt if user does not input a valid choice."""
    text += " [" + "/".join(choices) + "]"
    while True:
        inp = input(text)
        if inp in choices:
            return inp

def shell_cmd(*args):
    """Command must succeed for program to continue."""
    proc = subprocess.run(args)
    returncode = proc.returncode
    if returncode != 0:
        raise RuntimeError(
            f"Command {args} failed with return code {returncode}")
    return proc

SLIPPI_URL = "https://github.com/project-slippi/Ishiiruka/releases/download/v2.2.3/FM-Slippi-2.2.3-Mac.zip"
GALE_URL = "https://raw.githubusercontent.com/altf4/slippi-ssbm-asm/libmelee/Output/Netplay/GALE01r2.ini"

@click.command()
def main():
    iso = input("Path to ISO: ")
    if not Path(iso).exists():
        print(f"WARNING: {iso} does not exist.")
    if Path("./out").exists():
        if prompt(f"./out ({Path('./out').resolve()}) exists, delete?", ["Y", "n"]) == "Y":
            shell_cmd("rm", "-rvf", "./out")
        else:
            print("abort")
            sys.exit(1)
    shell_cmd("mkdir", "out")
    shell_cmd("wget", SLIPPI_URL, "-O", "out/slippi.zip")
    shell_cmd("unzip", "out/slippi.zip")
    shell_cmd("rm", "out/slippi.zip")
    shell_cmd("mv", "Slippi Dolphin.app", "out/dolphin-emu.app")
    shell_cmd("rm", "-rvf", "__MACOSX")
    shell_cmd("git", "clone", "--branch", "dev", "git@github.com:18praveenb/libmelee.git", "out/libmelee")
    shell_cmd("touch", "out/dolphin-emu.app/Contents/MacOS/portable.txt")
    shell_cmd("wget", GALE_URL, "-O", "out/dolphin-emu.app/Contents/Resources/Sys/GameSettings/GALE01r2.ini")
    shell_cmd("mkdir", "-p", "out/dolphin-emu.app/Contents/Resources/User/Pipes")
    shell_cmd("open", "out/dolphin-emu.app")
    time.sleep(2)
    example_py_cmd = ["out/libmelee/example.py", "--dolphin_executable", "out/dolphin-emu.app/Contents/MacOS", "--iso", iso]
    print("Success. To run the example, quit the Dolphin emulator")
    print("   that was launched, and run the command:")
    print(" ".join(example_py_cmd))
    

if __name__ == "__main__":
    main()
