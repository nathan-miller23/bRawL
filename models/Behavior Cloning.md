# How to run a behavior cloning agent

## Collect data

```
python3 path/to/SSBMEnv.py -i path/to/iso -e path/to/dolphin-emu.app/Contents/MacOS -l cpu-level -m -s
```

The data is output to `out_1.txt`, `out_2.txt` etc. It's actually a JSON but with txt extension (oops)

**Make sure to set -sp to some value if you're starting another one**.
The numbering resets and it will overwrite the previous run otherwise.

## Train BC agent

See `bc.ipynb`

by default it loads `testdump_1.txt`, trains and saves a state dict to `bc_agent`

Note: You might run into an issue where some of the JSON data
has state, but not next_state or action. Just delete these data.
(And ideally push the fix to bc.ipynb)

## Eval BC agent

`bc.ipynb` also has code to run it in the gym environment against a CPU.