"""
Dynaface measures facial symmetry and movement in photos and videos.

Load the AI models with dynaface.models.init_models(), then analyze a face with
dynaface.facial.load_face_image() or dynaface.facial.AnalyzeFace.
"""

try:
    from importlib.metadata import version

    __version__ = version("dynaface")
except:
    __version__ = "unknown"
