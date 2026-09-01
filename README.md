# MLBFM

The implementation for method proposed in Multimodal LLM for Function-aware Building Polygon Extraction from Remote Sensing Imagery.

![model](readme/1a.png)

## 1. Data Preparation

Please refer to [`README/DATA.md`](README/DATA.md) for detailed instructions on dataset preparation.

## 2. Environment Setup

Please refer to [`README/INSTALL.md`](README/INSTALL.md) for instructions on installing and configuring the required environment.

## 3. Training

To train MLBFM on your own device and datasets, follow the steps below.

### Step 1: Configure the Dataset

Open:


MLBFM_ROOT/src/lib/datasets/dataset/BuildFunc.py
````

Modify the following parameters according to your dataset:

* `num_classes`
* `train_path`
* `val_path`

### Step 2: Configure the DANCE Module

Open:

```text
MLBFM_ROOT/src/lib/models/dance_lib/networks/dance/evolve813.py
```

Set:

```python
class_num = 10
```

> Set `class_num` to the number of building-function categories in your dataset.

### Step 3: Configure the ResNet Backbone

Open:

```text
MLBFM_ROOT/src/lib/models/networks/msra_resnetv4.py
```

Set:

```python
class_num = 10
```

> The value of `class_num` should be consistent with the number of building-function categories in your dataset.

### Step 4: Configure Building-Function Categories

Open:

```text
MLBFM_ROOT/src/lib/detectors/ctdet.py
```

Modify the `building_functions` list according to your dataset. For example:

```python
building_functions = [
    'dense residential',
    'business',
    'commercial',
    'residential',
    'factory',
    'government',
    'hospital',
    'resort',
    'public',
    'school'
]
```

> The order of the categories should be consistent with the class indices used in your dataset.

### Step 5: Configure Building-Function Categories in the Trainer

Open:

```text
MLBFM_ROOT/src/lib/trains/base_trainer.py
```

Modify the `building_functions` list so that it is consistent with the category definition in Step 4.

For example:

```python
building_functions = [
    'dense residential',
    'business',
    'commercial',
    'residential',
    'factory',
    'government',
    'hospital',
    'resort',
    'public',
    'school'
]
```

### Step 6: Configure Category Weights

Open:

```text
MLBFM_ROOT/src/lib/trains/ctdet.py
```

Modify:

```python
build_ce_with_fixed_weights
```

Set the category weights according to the class distribution of your dataset.

> Appropriate class weights are recommended when the building-function categories are significantly imbalanced.

### Step 7: Configure Training Parameters

Open:

```text
MLBFM_ROOT/src/lib/opts.py
```

Modify the training parameters according to your requirements, including:

* `exp_id`
* `lr_step`
* `num_epochs`
* `lr`
* `batch_size`
* and other relevant parameters

### Step 8: Start Training

Open a terminal in the MLBFM root directory:

```bash
cd MLBFM_ROOT
```

Then run:

```bash
python src/main.py
```

The model will start training using the configured dataset and parameters.

---

## 4. Testing

To perform inference on your own device and datasets, follow the steps below.

### Step 1: Configure the Inference Script

Open:

```text
MLBFM_ROOT/src/preditc_rs2_buff.py
```

Modify the following parameters:

* `model_path`: path to the trained model checkpoint
* `save_path`: directory for saving the inference results
* `image_path`: path to the input image or dataset

### Step 2: Run Inference

Open a terminal in the `src` directory:

```bash
cd MLBFM_ROOT/src
```

Then run:

```bash
python preditc_rs2_buff.py
```

The model will perform inference on the specified images and save the results to `save_path`.

```
```
