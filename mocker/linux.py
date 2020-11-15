#!/usr/bin/env python3
import subprocess
from pathlib import Path
import shutil
import logging
import sys
import time
import click
from shared import urls, prompt, shell_cmd, install_libmelee

@click.command()
def main():
    iso = input("Path to ISO: ")
    if not Path(iso).exists():
        print(f"WARNING: {iso} does not exist.")
        if prompt("Continue?", ["y", "n"]) == "n":
            print("abort")
            sys.exit(1)
    if Path("./dolphin-emu").exists():
        if prompt(f"./dolphin-emu ({Path('./dolphin-emu').resolve()}) exists, delete?", ["y", "n"]) == "y":
            shell_cmd("rm", "-rvf", "./dolphin-emu")
        else:
            print("abort")
            sys.exit(1)
    if not install_libmelee():
        if prompt("Libmelee install failed, quit installer?", ["Y", "n"]) == "Y":
            print("abort")
            sys.exit(1)
    shell_cmd("wget", urls["slippi-linux"], "-O", "./slippi.AppImage")
    shell_cmd("chmod", "+x", "./slippi.AppImage")
    shell_cmd("./slippi.AppImage", "--appimage-extract")
    shell_cmd("rm", "./slippi.AppImage")
    shell_cmd("mv", "./squashfs-root/usr", "./dolphin-emu")
    shell_cmd("rm", "-rvf", "squashfs-root")
    shell_cmd("touch", "./dolphin-emu/bin/portable.txt")
    shell_cmd("wget", urls["gale"], "-O", "./dolphin-emu/bin/Sys/GameSettings/GALE01r2.ini")
    print("Quit the dolphin emulator that opens")
    shell_cmd("./dolphin-emu/bin/dolphin-emu")
    time.sleep(2)
    example_py_cmd = ["python", "../libmelee/example.py", "--dolphin_executable", "./dolphin-emu/bin/dolphin-emu", "--iso", iso]
    print("Success. To run the example, quit the Dolphin emulator")
    print("   that was launched, and run the command:")
    print(" ".join(example_py_cmd))
    

if __name__ == "__main__":
    main()
