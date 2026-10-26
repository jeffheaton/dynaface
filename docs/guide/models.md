# Models and devices

Dynaface uses three AI models, which it runs with [ONNX Runtime](https://onnxruntime.ai):

| Model | What it does | License |
|---|---|---|
| [BlazeFace](https://doi.org/10.48550/arXiv.1907.05047) (short range) | Finds the face | Apache 2.0 (MediaPipe) |
| [SPIGA](https://doi.org/10.48550/arXiv.2210.07233), trained on WFLW | Places the 98 landmarks and estimates the head pose | BSD 3-Clause |
| [U²-Net](https://doi.org/10.1016/j.patcog.2020.107404) | Removes the background from profiles | Apache 2.0 |

## Downloading the models

[`download_models()`][dynaface.models.download_models] downloads the models, about 400 MB as one zip file, checks it against a SHA-256 hash and unpacks it to `~/.dynaface/models`. If the models are already there, it just returns the folder. To keep them somewhere else, pass a folder:

```python
from dynaface import models

path = models.download_models("/data/dynaface-models")
models.init_models(path, models.detect_device())
```

### Without internet access

On a computer that can't download the models, copy the models folder from another computer, or download the [model bundle](https://data.heatonresearch.com/dynaface/model/2/dynaface_models.zip) and unzip it. The folder should hold the three `.onnx` files. Then give the folder to `init_models()`.

### On networks that inspect HTTPS

Some corporate networks intercept HTTPS traffic with their own certificate, which makes the download fail with a certificate error. Point Python's `requests` library at your organization's certificate bundle with the `REQUESTS_CA_BUNDLE` environment variable, or download the bundle in a browser and unzip it as above.

## Choosing a device

[`detect_device()`][dynaface.models.detect_device] picks the fastest device ONNX Runtime can use: `"cuda"` when an NVIDIA GPU is available, `"mps"` (Core ML) on a Mac that supports it, and `"cpu"` otherwise. To choose for yourself, pass one of these to [`init_models()`][dynaface.models.init_models]. A GPU matters most for video. Using `"cuda"` needs ONNX Runtime's CUDA build, the `onnxruntime-gpu` package, and NVIDIA's CUDA libraries.

[`unload_models()`][dynaface.models.unload_models] releases the models. Call `init_models()` again before analyzing more faces.

## Citing the models

If you publish results, please cite the models along with Dynaface:

- Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2019). BlazeFace: Sub-millisecond neural face detection on mobile GPUs. *CVPR Workshop on Computer Vision for AR/VR*. <https://doi.org/10.48550/arXiv.1907.05047>
- Prados-Torreblanca, A., Buenaposada, J. M., & Baumela, L. (2022). Shape preserving facial landmarks with graph attention networks. *British Machine Vision Conference (BMVC)*. <https://doi.org/10.48550/arXiv.2210.07233>
- Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020). U²-Net: Going deeper with nested U-structure for salient object detection. *Pattern Recognition, 106*, 107404. <https://doi.org/10.1016/j.patcog.2020.107404>
