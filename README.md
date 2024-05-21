# TROVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks :hammer_and_wrench:

## Setup

Install the required packages:

```python
pip install -r requirements.txt
```

Tasks and datasets are organized as follows:
```
├── MATH
│   ├── algebra
│   ├── counting_and_probability
│   ├── geometry
│   ├── intermediate_algebra
│   ├── number_theory
│   ├── prealgebra
│   └── precalculus
├── TableQA
│   ├── TabMWP
│   ├── WTQ
│   └── HiTab
├── VQA
└── └── GQA
```

## Running Experiments

### Our Method: TroVE

```bash
python run_trove.py --task_name "math/algebra"
```

* For MATH tasks, specify the task name as _math/${dataset_name}_, e.g., _math/algebra_.
* For TableQA and VQA tasks, directly used the dataset name: [_tabmwp_, _wtq_, _hitab_, _gqa_].

Note that the specified `--task_name` argument should be lowercased.

### Baseline Methods: Primitive & Instance

```bash
python baseline.py --task_name "math/algebra" --suffix "primitive"  # or "instance"
```

Note that for GQA dataset, we implement the `locate_objects` and `visual_qa` functions as fast apis.
So you need to launch the server first (as below), then run the trove/baseline experiments.
启动1个图像问答和图像定位的API接口，toolbox/gqa.py中写了2个工具，可以使用这些接口
```bash 
uvicorn server.gqa:app
```

## Evaluation

```python
python -m utils.eval --results_path ${RESULTS_PATH}
```


## 工具箱中每个工具的格式
#%%开头表示是工具的描述
然后下面是工具的主要内容
```python
# %% Locate object bounding boxes in the image
import requests
from PIL import Image

def locate_objects(image: str | Image.Image, object_name: str) -> list:
    """Load object bounding boxes in the image.
    Args:
        image: str, file name of the image
        object: str, natural language description of the object
    Rets:
        selected_boxes: box regions of the found object(s)
    """
    params = {"object_name": object_name}

    if not isinstance(image, str):
        image.save("tmp.jpg")
        params["image_name"] = "tmp.jpg"
    else:
        params["image_name"] = image

    r = requests.get(
        "http://127.0.0.1:8000/loc", params=params
    )
    return r.json()["boxes"] 

```