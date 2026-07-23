# Dynaface

Dynaface is a computer application and Python library that measures facial symmetry. It utilizes advanced AI for both symmetric and asymmetric faces. AI allows Dynaface to locate 97 facial landmarks automatically to conduct a variety of measures on still images and videos. Researchers can export all data from pictures and videos to CSV/Excel format for further analysis.

[The story behind Dynaface, from RGA](https://bit.ly/4201i3j)

![Dynaface Screenshot](https://s3.amazonaws.com/data.heatonresearch.com/images/facial/site/dynaface-1.jpg?v=1)

# Helpful Links

- [Dynaface iPhone and iPad](https://apps.apple.com/us/app/dynaface/id6475224118)
- [Download Latest Version](https://github.com/jeffheaton/dynaface/releases/tag/v1.3.0) or [All Versions](https://github.com/jeffheaton/dynaface/releases)
- [Users Guide](https://github.com/jeffheaton/dynaface/blob/main/dynaface-app/manual.md)
- Dynaface Library: [Python Library](https://github.com/jeffheaton/dynaface/tree/main/dynaface-lib-python), [DotNet Library](https://github.com/jeffheaton/dynaface/tree/main/dynaface-lib-dotnet)

## Neural Network Models

Dynaface builds on three neural networks, all run via ONNX Runtime:

| # | Model | Role in Dynaface | File | License |
|---|-------|------------------|------|---------|
| 1 | **BlazeFace** (short-range) | Face bbox detection | `blaze_face_short_range.onnx` | Apache-2.0 (MediaPipe) |
| 2 | **SPIGA** (WFLW-98) | 98-pt landmarks + head pose | `spiga_wflw.onnx` | BSD-3-Clause |
| 3 | **U²-Net** | Background/saliency removal (lateral) | `u2net.onnx` | Apache-2.0 |

You can download all three models from the [current model bundle](https://data.heatonresearch.com/dynaface/model/2/dynaface_models.zip).

If you use Dynaface in academic work, please also cite their original authors:

- **BlazeFace** (face detection) — Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2019). BlazeFace: Sub-millisecond neural face detection on mobile GPUs. _CVPR Workshop on Computer Vision for AR/VR_. https://doi.org/10.48550/arXiv.1907.05047
- **SPIGA** (facial landmarks & head pose) — Prados-Torreblanca, A., Buenaposada, J. M., & Baumela, L. (2022). Shape preserving facial landmarks with graph attention networks. _British Machine Vision Conference (BMVC)_. https://doi.org/10.48550/arXiv.2210.07233
- **U²-Net** (background removal) — Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020). U²-Net: Going deeper with nested U-structure for salient object detection. _Pattern Recognition, 106_, 107404. https://doi.org/10.1016/j.patcog.2020.107404

## To Cite Dynaface

- Renne, A., Heaton, J., & Boahene, K. D. O. (2026). Associations of AI-based facial metrics with patient-reported outcomes in idiopathic facial paralysis. _Laryngoscope_. Advance online publication. https://doi.org/10.1002/lary.70417
- BibTeX: https://github.com/jeffheaton/dynaface/blob/main/CITATION.bib

## Other Works

- Renne, A., Heaton, J., Derakhshan, A., Nellis, J. C., Desai, S. C., & Boahene, K. D. (2025). Use of dynamic, automated facial analysis in quantifying oral-ocular synkinesis. _Facial Plastic Surgery & Aesthetic Medicine_. https://doi.org/10.1177/26893614251395737

- Berges, A. J., Renne, A., Heaton, J., Leung, D. G., & Boahene, K. D. (2025). Facial weakness in facioscapulohumeral muscular dystrophy: Objective and patient-reported measures to guide reconstructive interventions. _Facial Plastic Surgery & Aesthetic Medicine_, Article 26893614251407675. https://doi.org/10.1177/26893614251407675

# License

This application and Python library are licensed under the [Apache License Version 2.0](https://github.com/jeffheaton/dynaface/blob/main/LICENSE.txt).
