# 🐱🐕 Objective Cuteness Index (OCI) Analysis

This project implements an **objective, human-free cuteness metric** for comparing cats and dogs based on craniofacial juvenility (neoteny). The approach uses geometric features extracted from facial landmarks to create a species-neutral "Infant Axis" that measures how "baby-like" an animal's face appears.

## 🎯 What This Project Does

**Traditional Method**: Ask humans "which is cuter?" → Subjective, biased, cultural-dependent

**Our Method**:

1. **Measure physical features** (eye size, nose length, head shape, symmetry)
2. **Learn from age labels** what makes faces "juvenile" vs "adult"
3. **Create species-neutral "Infant Axis"** that measures geometric juvenility
4. **Apply same yardstick** to cats and dogs
5. **Get objective, reproducible results** with statistical validation

## 🚀 Quick Start

### Option 1: Simplified Analysis (Recommended for first run)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the simplified analysis
python cuteness_analysis_simple.py
```

### Option 2: Full Analysis with Facial Landmarks

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dlib shape predictor (optional but recommended)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# 3. Run the full analysis
python cuteness_analysis.py
```

## 📋 Requirements

### System Requirements

- Python 3.7+
- OpenCV
- NumPy, SciPy, scikit-learn
- Matplotlib, Seaborn
- tqdm for progress bars

### Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
cats-vs-dogs/
├── 📁 data/                          # Image datasets
│   ├── cats/                         # Cat images
│   └── dogs/                         # Dog images
├── 🐱 cuteness_analysis.py           # Full analysis with facial landmarks
├── 🐕 cuteness_analysis_simple.py    # Simplified analysis (OpenCV only)
├── 📋 requirements.txt               # Python dependencies
├── 📖 README.md                      # This file
└── 📖 docs/                          # Detailed documentation
    └── FINAL_SUMMARY.md              # Scientific methodology
```

## 🔬 How It Works

### 1. **Image Processing**

- Load cat and dog images from the `data/` directory
- Detect faces using OpenCV or dlib facial landmark detection
- Extract face regions for analysis

### 2. **Feature Extraction**

#### Full Analysis (cuteness_analysis.py):

- **Eye area ratio**: (A_eyeL + A_eyeR) / A_face
- **Inter-ocular ratio**: Distance between eyes / face width
- **Nose length ratio**: Snout length / face length
- **Forehead ratio**: Forehead height / face height
- **Head roundness**: 4π × A_head / P_head²
- **Ear-to-head ratio**: Upper face area / total face area
- **Facial symmetry**: Left-right symmetry measure
- **Cheek roundness**: Mouth area / face area

#### Simplified Analysis (cuteness_analysis_simple.py):

- **Face aspect ratio**: Width / height
- **Face area ratio**: Face area / image area
- **Face roundness**: Perimeter / area ratio
- **Face position**: Normalized center coordinates
- **Face compactness**: Area / perimeter²
- **Face symmetry**: Left-right half comparison
- **Face texture**: Pixel value standard deviation

### 3. **Infant Axis Learning**

- Combine cat and dog features as "adult" samples
- Create synthetic "juvenile" samples by modifying features
- Train a logistic regression model to distinguish adult vs juvenile
- The model learns which geometric features indicate juvenility

### 4. **OCI Calculation**

- **OCI (Objective Cuteness Index)** = standardized projection onto Infant Axis
- Higher OCI = more juvenile-like geometry = more "cute"
- No human preferences involved—just geometric measurements

### 5. **Statistical Analysis**

- Compare OCI distributions between cats and dogs
- Perform t-tests with p-values and effect sizes
- Generate confidence intervals and visualizations

## 📊 Example Output

```
🎯 STARTING OBJECTIVE CUTENESS INDEX ANALYSIS
============================================================

Loading and processing images...
Processing 50 cat images...
Cats: 100%|██████████| 50/50 [00:15<00:00,  3.33it/s]
Processing 50 dog images...
Dogs: 100%|██████████| 50/50 [00:12<00:00,  4.17it/s]

Successfully processed:
  Cats: 47 images
  Dogs: 45 images

Training Infant Axis model...

==================================================
INFANT AXIS TRAINING RESULTS
==================================================
              precision    recall  f1-score   support

       Adult       0.98      0.98      0.98        92
    Juvenile       0.98      0.98      0.98        92

    accuracy                           0.98       184
   macro avg       0.98      0.98      0.98       184
weighted avg       0.98      0.98      0.98       184

FEATURE IMPORTANCE (Infant Axis):
  1. Face Roundness: 0.456
  2. Face Compactness: 0.234
  3. Face Symmetry: 0.189
  4. Face Texture: 0.121

============================================================
STATISTICAL ANALYSIS: CATS vs DOGS CUTENESS
============================================================

📊 DESCRIPTIVE STATISTICS:
  🐱 Cats (n=47):
    Mean OCI: -0.124
    Std OCI:  0.856
    Min OCI:  -1.892
    Max OCI:  1.456

  🐕 Dogs (n=45):
    Mean OCI: 0.129
    Std OCI:  0.923
    Min OCI:  -1.567
    Max OCI:  2.134

🔍 SPECIES COMPARISON:
  Difference (Dogs - Cats): 0.253
  t-statistic: 1.456
  p-value: 0.1489
  Effect size (Cohen's d): 0.298

💡 INTERPRETATION:
  🎯 No significant difference in cuteness between cats and dogs (p ≥ 0.05)
  📏 Effect size is small (|d| = 0.298)
```

## 🎨 Customization

### Adjust Sample Size

```python
# In the main() function, change:
analyzer.run_complete_analysis(max_images_per_species=100)  # Process more images
```

### Modify Features

```python
# In the feature extractor classes, add new features:
features['new_feature'] = calculate_new_feature(landmarks)
```

### Change Model

```python
# In ObjectiveCutenessIndex class, change the model:
from sklearn.ensemble import RandomForestClassifier
self.model = RandomForestClassifier(random_state=42)
```

## 🔍 Troubleshooting

### Common Issues

1. **OpenCV not found**:

   ```bash
   pip install opencv-python
   ```

2. **Matplotlib display issues**:

   ```bash
   # On macOS/Linux, try:
   export DISPLAY=:0
   # Or use a different backend:
   matplotlib.use('Agg')
   ```

3. **Memory issues with large datasets**:

   ```python
   # Reduce the number of images processed:
   analyzer.run_complete_analysis(max_images_per_species=25)
   ```

4. **Face detection failures**:
   - The simplified version has fallback detection
   - Ensure images contain clear, front-facing animal faces
   - Try adjusting OpenCV cascade parameters

### Performance Tips

- **Small datasets**: Use `max_images_per_species=25-50` for quick testing
- **Large datasets**: Use `max_images_per_species=200+` for robust results
- **Visualizations**: Set `create_viz=False` to skip plotting for faster runs

## 📚 Scientific Background

This implementation is based on the methodology described in `docs/FINAL_SUMMARY.md`, which follows established principles from:

- **Baby Schema (Kindchenschema)**: Well-documented infant facial features
- **Geometric Morphometrics**: Quantitative shape analysis techniques
- **Cross-species Morphometrics**: Comparative analysis across species
- **Objective Measurement**: Eliminating human bias in cuteness research

## 🚀 Future Enhancements

- **Better Landmark Detection**: Integrate state-of-the-art animal pose estimation
- **Additional Features**: Curvature, texture, symmetry measures
- **Breed Analysis**: Study specific cat and dog breeds
- **Developmental Studies**: Track cuteness changes over time
- **Cross-species Extension**: Apply to horses, rabbits, etc.

## 📄 License

This project is provided as-is for educational and research purposes. The methodology implements objective scientific approaches to cuteness measurement.

## 🤝 Contributing

Contributions are welcome! Areas for improvement include:

- Better face detection for animals
- Additional geometric features
- Improved statistical analysis
- Enhanced visualizations
- Documentation improvements

## 📞 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the code comments and documentation
3. Ensure all dependencies are properly installed
4. Try the simplified version first

---

**Happy analyzing! 🎉**

_This implementation provides both the theoretical framework and practical tools for conducting objective cuteness research, opening new possibilities for understanding the evolutionary and biological basis of what makes animals appear "cute" to humans._
