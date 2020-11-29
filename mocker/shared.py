import subprocess

urls = {
    'gale': "https://raw.githubusercontent.com/altf4/slippi-ssbm-asm/libmelee/Output/Netplay/GALE01r2.ini",
    'slippi-mac': "https://github.com/project-slippi/Ishiiruka/releases/download/v2.2.3/FM-Slippi-2.2.3-Mac.zip",
    'slippi-linux': "https://github.com/project-slippi/Ishiiruka/releases/download/v2.2.3/Slippi_Online-x86_64.AppImage"
}

def prompt(text, choices):
    """Will repeat prompt if user does not input a valid choice."""
    text += " [" + "/".join(choices) + "] "
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

def install_libmelee():
    print("Will now install libmelee")
    print("Make sure you are in a virtual environment and already ran git "
          + "submodule update --init and pip install -r requirements.txt")
    if prompt(f"Ok to continue?", ["y", "n"]) == "y":
        subprocess.run(["python", "-m", "pip", "install", "-e", "../libmelee"])
        return True
    return False