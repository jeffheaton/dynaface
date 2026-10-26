# Security policy

## Supported versions

Security fixes go into the latest release of each part of Dynaface: the Python library (`dynaface` on PyPI), the .NET library (`Dynaface` and `Dynaface.Onnx` on NuGet) and the Dynaface app. Please check that a problem still happens on the latest release before reporting it.

| Version | Supported |
|---|---|
| 2.0.x | Yes |
| Older | No |

## Reporting a vulnerability

Please **don't report security problems in public issues**. Email jeff@jeffheaton.com instead, and include:

- which part of Dynaface is affected, and its version;
- what the problem is and what someone could do with it;
- the steps to reproduce it, or a small proof of concept.

Don't include photos, videos or data of identifiable patients; see [Protect patient privacy](CONTRIBUTING.md#protect-patient-privacy).

The maintainer will confirm the report and keep you updated while it's fixed. Please don't disclose the problem publicly until a fix is released. If you'd like, you'll be credited in the release notes.

## What to report

Dynaface runs entirely on your own computer or device, and it doesn't send images or measurements anywhere (see the [privacy policy](dynaface-privacy.md)). The Python library only goes online to download its ONNX models once, which it checks against a SHA-256 hash, and to load an image from a URL when you ask it to. The .NET library doesn't download anything; you supply the models.

Examples of what to report:

- a crafted image, video or file that makes Dynaface run code, or read or write files it shouldn't;
- anything that could expose patient images or measurements;
- a way to tamper with the model download.

Problems in third-party packages such as OpenCV, ONNX Runtime or PyQt belong with those projects, unless Dynaface uses them in an unsafe way.

## Project files

Dynaface 2.0.3 and earlier save desktop-app projects (`.dyfc` files) with Python's `pickle` format, and opening a pickle file can run code stored in it. If you use one of those versions, only open `.dyfc` files that you created yourself or that come from someone you trust.

Later versions save projects as a zip of JSON and numpy arrays, which can't run code when opened. They still open older project files, but they accept only the plain data those files hold and refuse anything else before it can run.
