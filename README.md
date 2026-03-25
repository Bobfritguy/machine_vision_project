# Event-Based ASL Gesture Recognition

Real-time American Sign Language recognition using a Prophesee GenX320 event camera. The classifier runs directly on the event stream and displays predictions through a browser-based dashboard.

The system was trained on the [ASL-DVS dataset](https://github.com/PIX2NVS/NVS2Graph) (captured with a DAVIS240C sensor) and optionally supplemented with recordings taken directly on the GenX320.


## What is an event camera?

A standard camera captures full frames at a fixed rate (e.g. 30 FPS). An event camera works differently: each pixel fires independently the moment its brightness changes, producing a stream of events `(t, x, y, p)` where `t` is a microsecond timestamp, `x` and `y` are pixel coordinates, and `p` is polarity (whether brightness increased or decreased). This gives very high temporal resolution with almost no motion blur and very low latency, but requires a different processing approach than frame-based vision.

The Prophesee GenX320 is a 320x320 pixel event camera based on the IMX636 sensor.


## Project layout

```
model.py            Neural network definition  
utils.py            Shared config dataclass, class labels, helpers  
preprocess.py       Converts raw event arrays to tensors the model can ingest  
dataset.py          PyTorch Dataset classes for loading training data  
train.py            Training script  
infer\_live.py       Core inference class (wraps model + preprocessing)  
infer\_web\_live.py   Flask web dashboard for live inference  
record\_data.py      Tool for recording labelled training samples from the camera  
biases/             Sensor bias configuration files (JSON)  
runs/               Training outputs (checkpoints, history, config)  
datasets/           Training data
```


## Setup

Python 3.11 is recommended.

```
pip install -r requirements.txt
```

The Metavision SDK must be installed separately from Prophesee to access the camera. It is not available via pip and requires a Prophesee account.


## How the pipeline works

### 1. Events to tensors (`preprocess.py`)

Raw events cannot be fed directly to a convolutional network. `events\_to\_frame` converts a window of events into a fixed-size tensor using one of three representations:

**`two\_channel`** (default): Creates a 2-channel image where channel 0 accumulates ON events (polarity 1) and channel 1 accumulates OFF events (polarity 0). Each pixel value is the log1p-normalised count of events at that location. Simple and retains full spatial information, but discards all temporal ordering within the window.

**`signed`**: A single-channel frame where each pixel value is (ON count - OFF count). Compact, but loses the distinction between a high-activity pixel where ON and OFF cancel out versus a low-activity pixel (both appear near zero).

**`voxel\_grid`**: Divides the time window into N temporal bins and produces one channel per bin. Preserves the order of events within the window. More expressive but slower and uses more memory. Note: ASL-DVS timestamps are in milliseconds, while GenX320 timestamps are in microseconds. For `two\_channel` and `signed` this makes no difference since timestamps are ignored, but it affects voxel bin boundaries.

**Resolution remapping:** The ASL-DVS dataset was captured at 120x90. The GenX320 is 320x320. Before accumulation, event coordinates are scaled from the source resolution to the target resolution (256x256). The strategy `accumulate\_then\_resize` accumulates events at native source resolution first, then resizes the resulting frame with bicubic interpolation. This preserves spatial density better than remapping sparse point coordinates directly.

### 2. Model (`model.py`)

MobileNetV3-Small pretrained on ImageNet, with two modifications:

- The first convolution layer is replaced to accept a variable number of input channels (2 for `two\_channel`, 1 for `signed`, N for `voxel\_grid`) rather than the default 3 (RGB).

- The classifier head is replaced with a linear layer outputting 24 classes (the 24 static ASL letters; J and Z are excluded from ASL-DVS because they require motion).

MobileNetV3-Small was chosen for low inference latency.

### 3. Training (`train.py`)

```
python train.py \\  
    --root datasets/ASL \\  
    --repr two\_channel \\  
    --target-size 256 \\  
    --epochs 30 \\  
    --out runs/my\_run
```

Key flags:

| Flag | Default | Notes |
| - | - | - |
| `--repr` | `two\_channel` | Event representation: `two\_channel`, `signed`, or `voxel\_grid` |
| `--target-size` | `224` | Square side length of the tensor fed to the model |
| `--lr` | `1e-3` | Learning rate (AdamW optimiser, cosine annealing schedule) |
| `--batch-size` | `64` |  |
| `--no-augment` | off | Disables all augmentation |
| `--extra-data` | None | Path to a flat dataset directory (e.g. GenX320 recordings) |
| `--oversample-factor` | auto | How much more likely a GenX320 sample is to be drawn per step |


**Data augmentation** is applied only during training to bridge the gap between the DAVIS240C (training sensor) and GenX320 (inference sensor). `EventFrameAugment` applies a random subset of transforms each pass:

- Vertical flip (50%) to allow the hand to enter from the other side of the frame.

- Rotation up to 30 degrees + translation (50%)

- Scale and aspect ratio jitter (50%)

- Contrast scaling (50%)

- Event dropout: randomly zeroes out a fraction of active pixels (50%)

- Cutout: blacks out a random rectangular patch (30%)

- Gaussian noise (50%)

**Mixed dataset training** uses `WeightedRandomSampler` to oversample the smaller dataset so both contribute roughly equally per epoch. The auto factor is `n\_asl / n\_extra`. With 80,640 ASL-DVS samples and 2,420 GenX320 samples the auto factor is ~33, meaning each GenX320 sample is drawn approximately 17 times per epoch (with different random augmentations applied each time, so no two draws are identical).

### 4. Inference

The `LiveInferencer` class in `infer\_live.py` handles the mismatch between training conditions and the live camera:

**Aspect ratio:** The ASL-DVS dataset was captured at 4:3 (240x180). The GenX320 is 1:1 (320x320). By default (`crop\_to\_training\_aspect=True`) events from the bottom quarter of the frame are discarded, leaving a 320x240 region matching the training sensor's aspect ratio. The source resolution passed to preprocessing is then (320, 240), not (320, 320).

**Y-axis orientation:** The DAVIS240C and GenX320 use opposite y-axis conventions. Events from the GenX320 appear upside-down relative to training data. The y coordinate is flipped before preprocessing (`flip\_y=True`).

**Sliding window:** The camera is read in 50ms chunks for a responsive display (~20 FPS). A rolling 200ms buffer feeds inference. The model runs on the full 200ms window each chunk.

**EMA smoothing:** Raw prediction probabilities are blended with previous values using an exponential moving average (`alpha=0.4`). This prevents a single noisy window from flipping the displayed prediction. Lower alpha is smoother but slower to respond to a new gesture.

**Minimum event threshold:** If fewer than 500 events are in the window, inference is skipped to avoid spurious predictions when no hand is present.


## Web dashboard (`infer\_web\_live.py`)

```
python infer\_web\_live.py \\  
    --checkpoint runs/my\_run/checkpoint\_best.pt \\  
    --input-camera-config biases/natural\_sunlight\_bias.json \\  
    --stc-threshold-us 5000 \\  
    --afk-frequency 50 \\  
    --spelling-threshold 0.85
```

Open `http://\<host\>:5000` in a browser. If accessing from another machine on the network, use `--host 0.0.0.0`.

The dashboard shows:

- Live event feed with temporal decay (new events appear bright white, then fade to grey against black)

- Current prediction and confidence

- Top-K predictions with scores

- Inference latency and event rate

- Spelling recorder

**ROI (Region of Interest):** Click and drag on the live feed to draw a rectangle. Events outside the ROI are discarded before inference; events inside are remapped to fill the full 320x320 sensor space before being passed to the model. This isolates the hand from background noise. All events are still shown on the display and the ROI border is drawn as a green overlay. Click "Clear ROI" to remove it.

Why remap to full frame rather than just masking? The model expects a hand occupying a certain proportion of the frame (matching training data). Simply masking would leave most of the frame empty. Remapping scales the ROI content up to fill the frame, giving the model a view closer to training conditions.

**Spelling recorder:** Accumulates letters into words. A letter is committed when confidence exceeds `--spelling-threshold` and the same letter is predicted for a minimum streak of consecutive windows. Use the controls on the dashboard to start/stop recording and clear the buffer.

**Hardware filters:**

| Flag | What it does |
| - | - |
| `--stc-threshold-us` | Spatio-Temporal Contrast. Discards events not corroborated by a neighbour within this time window. Reduces noise. Try 5000-10000 us. |
| `--erc-rate` | Event Rate Control. Caps the maximum output rate (events/s). Useful under bright lighting. |
| `--afk-frequency` | Anti-Flicker. Suppresses events caused by mains lighting flicker. Use 50 for EU/UK, 60 for US. |


These require a direct HAL device connection and are unavailable in file playback mode.

**Full flag reference:**

| Flag | Default | Notes |
| - | - | - |
| `--checkpoint` | required | Path to trained model `.pt` file |
| `--repr` | `two\_channel` | Must match the representation the model was trained with |
| `--target-size` | `256` | Must match training |
| `--top-k` | `5` | Number of predictions shown in the sidebar |
| `--min-events` | `500` | Skip inference below this window event count |
| `--inference-window-us` | `200000` | Rolling inference window size in microseconds |
| `--smoothing-alpha` | `0.4` | EMA weight for new predictions (0 = ignore new, 1 = ignore history) |
| `--spelling-threshold` | `0.85` | Minimum confidence to commit a letter |
| `--no-crop-to-training-aspect` | off | Disables 4:3 crop (use only if model was trained on square data) |
| `--no-flip-y` | off | Disables y-axis flip |
| `--jpeg-quality` | `70` | Stream quality vs. bandwidth tradeoff |
| `--host` | `127.0.0.1` | Interface to bind to |
| `--port` | `5000` |  |



## Recording GenX320 training data (`record\_data.py`)

The ASL-DVS dataset was captured with a DAVIS240C. Recording data directly from the GenX320 and mixing it into training helps the model learn the characteristics specific to this sensor (noise profile, resolution, dynamic range).

```
python record\_data.py \\  
    --out datasets/genx320\_recorded \\  
    --input-camera-config biases/natural\_sunlight\_bias.json \\  
    --samples-per-burst 20 \\  
    --sample-duration-ms 200
```

The script cycles through all 24 letters. Controls:

| Key | Action |
| - | - |
| `SPACE` | Record a burst for the current letter |
| `ENTER` | Advance to the next letter |
| `\<letter\>` | Jump directly to that letter |
| `Q` | Quit and print session summary |


Each burst records `--samples-per-burst` samples of `--sample-duration-ms` milliseconds each with a short gap between them. Each sample is saved as a `.npy` file: a float32 array of shape `(N, 4)` with columns `\[t, x, y, p\]`, matching ASL-DVS format.

Output structure:

```
datasets/genx320\_recorded/  
    a/  
        a\_0001.npy  
        a\_0002.npy  
    b/  
        b\_0001.npy  
    ...
```

**Web viewfinder:** Add `--web` to open a browser preview at `http://localhost:62079` while recording. Shows the live event feed so you can confirm your hand is in frame and generating events before committing a burst.

**Recording tips:**

- Use a plain, static background. Background clutter generates events and adds noise to every sample.

- Small deliberate movements while holding the gesture generate more events than a completely still hand. The model expects some motion.

- The ASL-DVS dataset frames the hand slightly back from the camera, not tightly cropped. Matching that distance gives better results.

- Check the idle event rate before recording. With `natural\_sunlight\_bias.json` you should see roughly 10-20K events/s at rest. If it is higher, the bias thresholds need to be raised.

- Recording across different lighting conditions and hand positions improves generalisation.

**How many samples to record:** The ASL-DVS training set has ~80,000 samples. At 2,420 GenX320 samples the oversampling factor is ~33x, meaning each sample is drawn ~~**17 times per epoch. Augmentation varies each draw, but a larger pool is always better. **Due to time and personel constraints only 1,400 samples were created.


## Sensor bias tuning

The GenX320 bias values control pixel sensitivity. They are loaded from a JSON file with `--input-camera-config`.

```
\{  
  "ll\_biases\_state": \{  
    "bias": \[  
      \{"name": "bias\_diff",     "value": 51\},  
      \{"name": "bias\_diff\_off", "value": 45\},  
      \{"name": "bias\_diff\_on",  "value": 45\},  
      \{"name": "bias\_fo",       "value": 34\},  
      \{"name": "bias\_hpf",      "value": 40\},  
      \{"name": "bias\_refr",     "value": 28\}  
    \]  
  \}  
\}
```

The biases that matter most for noise control:

| Bias | Effect | Direction for less noise |
| - | - | - |
| `bias\_diff\_on` | Threshold for ON events (brightness increase) | Increase |
| `bias\_diff\_off` | Threshold for OFF events (brightness decrease) | Increase |
| `bias\_diff` | Overall contrast offset applied to both | Increase |
| `bias\_refr` | Refractory period: how long a pixel waits before firing again | Increase |


If the idle event rate (no movement) is above ~20K events/s, increase `bias\_diff\_on` and `bias\_diff\_off` by 2-3 units at a time and check again. If gestures generate too few events (model not triggering), decrease them. `bias\_refr` is a blunter tool: it limits how quickly any pixel can fire again, which helps with hot pixels (individual pixels that fire continuously regardless of scene content).

Included configs:

- `biases/natural\_sunlight\_bias.json`: `diff\_on=45, diff\_off=45, refr=28`. Target idle rate ~10-20K events/s.

- `biases/dark\_room\_bias.json`: Lower thresholds for low-light conditions.


## Checkpoints and saved config

Each training run saves to `--out`:

```
runs/my\_run/  
    checkpoint\_best.pt          Weights at best validation accuracy  
    checkpoint\_last.pt          Weights at final epoch  
    preprocess\_config.json      Preprocessing settings used during training  
    training\_history.json       Per-epoch loss and accuracy
```

`preprocess\_config.json` records the representation, source and target resolution, and normalisation strategy. These must match at inference time. If you train with `--repr two\_channel --target-size 256`, pass the same flags to `infer\_web\_live.py`.


## Dataset structure (ASL-DVS)

```
datasets/ASL/  
    SR\_Train/  
        LR/         120x90 (used for training by default)  
            a/  
                a\_0001.npy  
            ...  
        HR/         240x180 (higher resolution split, not used by default)  
    SR\_Test/  
        LR/  
        HR/
```

24 ASL static hand-shape letters (J and Z excluded because they involve motion). Each `.npy` file is a `(N, 4)` int32 array: `\[timestamp\_ms, x, y, polarity\]`. Approximately 3,360 samples per class in training (80,640 total) and 840 per class in test (20,160 total).

