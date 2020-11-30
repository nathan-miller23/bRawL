# How to run a behavior cloning agent

## Collect data

```
python3 path/to/SSBMEnv.py -i path/to/iso -e path/to/dolphin-emu.app/Contents/MacOS -l cpu-level -m -s path/to/dump/data
```

The data is output to `data_1.txt`, `data_2.txt` etc. It's actually a JSON but with txt extension (oops)

## Train BC agent

See `bc.ipynb`

by default it loads `testdump_1.txt`, trains and saves a state dict to `bc_agent`

## Eval BC agent

`bc.ipynb` also has code to run it in the gym environment against a CPU.