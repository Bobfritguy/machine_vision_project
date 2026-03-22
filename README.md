## To run

Install the requirements
```
pip install -r requirements.txt
```

Run the `infer_web_live.py` script. Info can be printed with:
```
python infer_web_live.py --help
```

Example command:
```
python infer_web_live.py --checkpoint models/model_with_augmentations.pt --input-camera-config biases/low_noise.json --stc-threshold-us 5000 --afk-frequency 50 --spelling-threshold 0.85
```


--- 

Some biases are in the `biases/` folder but they need to be tweaked and there is no "perfect" bias yet.

