#!/bin/bash

rm /ssbm/ssbm_gym/dolphin-exe/dolphin-emu
rm /ssbm/ssbm_gym/dolphin-exe/dolphin-emu-nogui
ln -s /dolphin/dolphin_src/build_nogui/Binaries/dolphin-emu-nogui /ssbm/ssbm_gym/dolphin-exe/
ln -s /dolphin/dolphin_src/build/Binaries/dolphin-emu /ssbm/ssbm_gym/dolphin-exe/

python3 -u /ssbm/ssbm_gym/test_env.py