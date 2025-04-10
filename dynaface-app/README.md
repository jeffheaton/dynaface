# Dynaface Python Application

Dynaface is an AI-driven facial tracking system utilizing computer vision techniques that integrate Convolutional Neural Networks (CNNs) with cascaded Graph Attention Network (GAT) regressors to enhance facial landmark detection by capturing both local appearance and global structural relationships (Zhou et al., 2022). This approach encodes both the appearance and spatial positioning of facial landmarks while employing an attention mechanism to prioritize reliable information. Such a method is particularly advantageous for assessing facial asymmetry in patients with facial paralysis, where conventional landmarking algorithms are often biased toward symmetric faces. By leveraging a global representation of facial structure, Dynaface enables precise detection of key landmarks despite asymmetries, facilitating the objective quantification of facial movement and asymmetry. These measurements, including the Facial Asymmetry Index (FAI) and Oral Commissure Excursion (OCE), serve as critical indicators of facial function and can be correlated with patient-reported outcomes to evaluate recovery and patient satisfaction.

# Helpful Python Commands

** Activate Environment **

```
source venv/bin/activate
.\venv\Scripts\activate.bat
.\venv\Scripts\Activate.ps1
```

** Allow Windows to Use Environment **

```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

** Run Unit Tests **

```
python -m unittest discover -s tests
```
