#!/usr/bin/env python3
import subprocess
from pathlib import Path
import sys
import time
import click
from shared import urls, prompt, shell_cmd, install_libmelee

@click.command()
@click.option("--iso")
def main(iso):
    if iso is None:
        iso = input("Path to ISO: ")
    if not Path(iso).exists():
        print(f"WARNING: {iso} does not exist.")
        if prompt("Continue?", ["y", "n"]) == "n":
            print("abort")
            sys.exit(1)
    if Path("./dolphin-emu.app").exists():
        if prompt(f"./dolphin-emu.app ({Path('./dolphin-emu.app').resolve()}) exists, delete?", ["y", "n"]) == "y":
            shell_cmd("rm", "-rvf", "./dolphin-emu.app")
        else:
            print("abort")
            sys.exit(1)
    install_libmelee()
    shell_cmd("wget", urls["slippi-mac"], "-O", "./slippi.zip")
    shell_cmd("unzip", "./slippi.zip")
    shell_cmd("rm", "./slippi.zip")
    shell_cmd("mv", "Slippi Dolphin.app", "dolphin-emu.app")
    shell_cmd("ln", "-s", "dolphin-emu.app/Contents/MacOS/Slippi Dolphin", "dolphin-emu.app/Contents/MacOS/dolphin-emu")
    shell_cmd("chmod", "+x", "dolphin-emu.app/Contents/MacOS/Slippi Dolphin", "dolphin-emu.app/Contents/MacOS/dolphin-emu")
    shell_cmd("rm", "-rvf", "__MACOSX")
    shell_cmd("touch", "./dolphin-emu.app/Contents/MacOS/portable.txt")
    shell_cmd("wget", urls["gale"], "-O", "./dolphin-emu.app/Contents/Resources/Sys/GameSettings/GALE01r2.ini")
    shell_cmd("mkdir", "-p", "./dolphin-emu.app/Contents/Resources/User/Pipes")
    shell_cmd("open", "./dolphin-emu.app")
    time.sleep(0.1)
    example_py_cmd = ["python", "../libmelee/example.py", "--dolphin_executable", "./dolphin-emu.app/Contents/MacOS", "--iso", iso]
    print("Success. To run the example, quit the Dolphin emulator")
    print("   that was launched, and run the command:")
    print(" ".join(example_py_cmd))
    

if __name__ == "__main__":
    main()
