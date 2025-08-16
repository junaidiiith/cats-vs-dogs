# Kaggle Datasets Guide for Cats vs Dogs Cuteness Analysis

This guide explains how to use the four Kaggle datasets available for the cuteness analysis project.

## 🎯 Available Datasets

### 1. **"borhanitrash/cat-dataset"** - Simple Cat Images

- **Description**: Collection of cat images without annotations
- **Structure**: Simple folder with `.jpg` files
- **Use Case**: Basic cat image analysis, feature extraction from images
- **Size**: ~268MB, multiple cat images

### 2. **"crawford/cat-dataset"** - Cat Images with Facial Landmarks

- **Description**: Cat images with corresponding `.cat` files containing facial landmark coordinates
- **Structure**: Multiple `CAT_XX` folders, each with images and `.cat` annotation files
- **Use Case**: Advanced facial analysis using geometric landmarks
- **Size**: ~4GB, organized by categories
- **Special Feature**: Facial landmark data for precise geometric measurements

### 3. **"jessicali9530/stanford-dogs-dataset"** - Dog Images with Breed Annotations

- **Description**: Stanford Dogs dataset with 120+ dog breeds and XML annotations
- **Structure**: Organized by breed with XML annotation files containing bounding boxes
- **Use Case**: Breed-specific analysis, detailed dog annotations
- **Size**: ~750MB, 120+ breeds
- **Special Feature**: Breed information and bounding box annotations

### 4. **"chetankv/dogs-cats-images"** - Combined Dogs and Cats Dataset

- **Description**: Balanced dataset with both dogs and cats, organized by species
- **Structure**: Training/test splits with separate cat and dog folders
- **Use Case**: Balanced species comparison, training/testing splits
- **Size**: ~435MB, balanced cat/dog distribution
- **Special Feature**: Pre-organized training/test splits

## 🚀 Quick Start

### Installation

```bash
cd cats_vs_dogs_cuteness
pip install -e .
```

### Basic Usage

```python
from cats_vs_dogs_cuteness import KaggleDatasetLoader

# Initialize the loader
loader = KaggleDatasetLoader()

# Get information about all datasets
dataset_info = loader.get_dataset_info()
print(dataset_info)
```

## 📊 Dataset Loading Methods

### 1. **Load Individual Datasets**

#### Dogs-Cats-Images Dataset

```python
# Load balanced dataset with specified samples per class
features, species, ages = loader.load_dogs_cats_images(
    max_samples_per_class=500  # 500 cats + 500 dogs = 1000 total
)

print(f"Loaded {len(features)} samples")
print(f"Cats: {species.count('cat')}")
print(f"Dogs: {species.count('dog')}")
```

#### Stanford Dogs Dataset

```python
# Load specific breeds with sample limits
features, species, ages = loader.load_stanford_dogs(
    max_breeds=20,           # Load 20 different breeds
    max_samples_per_breed=50 # 50 samples per breed
)

print(f"Loaded {len(features)} dog samples from {len(set(breed_labels))} breeds")
```

#### Crawford Cats Dataset

```python
# Load cat categories with facial landmarks
features, species, ages = loader.load_crawford_cats(
    max_categories=10,           # Load 10 CAT_XX categories
    max_samples_per_category=50  # 50 samples per category
)

print(f"Loaded {len(features)} cat samples with facial landmarks")
```

#### Borhanitrash Cats Dataset

```python
# Load simple cat images
features, species, ages = loader.load_borhanitrash_cats(
    max_samples=500  # Load up to 500 cat images
)

print(f"Loaded {len(features)} cat images")
```

### 2. **Load Combined Dataset**

```python
# Load from all available sources
features, species, ages = loader.load_combined_dataset(
    max_samples_per_dataset=200,  # 200 samples per dataset
    include_synthetic=True        # Generate synthetic data for failed loads
)

print(f"Combined dataset: {len(features)} total samples")
print(f"Cats: {species.count('cat')}")
print(f"Dogs: {species.count('dog')}")
```

## 🔍 Feature Extraction

### Image-Based Features

For datasets without landmarks (dogs-cats-images, borhanitrash-cats):

- **Face Detection**: Uses OpenCV Haar cascades
- **Geometric Features**: Aspect ratio, area ratio, edge density
- **Texture Features**: Variance, symmetry calculations
- **Fallback**: Synthetic features if face detection fails

### Landmark-Based Features

For datasets with landmarks (crawford-cats):

- **Facial Landmarks**: 18+ coordinate points from `.cat` files
- **Geometric Ratios**: Eye area, nose length, head roundness
- **Spatial Relationships**: Inter-ocular distance, facial symmetry
- **Precise Measurements**: Based on actual facial anatomy

### Feature Vector

All methods return 8-dimensional feature vectors:

1. `eye_area_ratio` - Relative eye size
2. `nose_length_ratio` - Nose proportion
3. `forehead_ratio` - Forehead prominence
4. `head_roundness` - Head shape circularity
5. `ear_to_head_ratio` - Ear size relative to head
6. `inter_ocular_ratio` - Eye spacing
7. `facial_symmetry` - Left-right symmetry
8. `cheek_roundness` - Cheek fullness

## 🧪 Testing the Loader

### Run Basic Tests

```bash
python test_kaggle_loader.py
```

### Run Comprehensive Example

```bash
python kaggle_example.py
```

## 📁 Dataset Structure Details

### Dogs-Cats-Images Structure

```
dataset/
├── training_set/
│   ├── cats/          # Cat images (.jpg)
│   └── dogs/          # Dog images (.jpg)
└── test_set/
    ├── cats/          # Test cat images
    └── dogs/          # Test dog images
```

### Stanford Dogs Structure

```
annotations/
└── Annotation/
    ├── n02085620-Chihuahua/     # Breed-specific folders
    ├── n02085782-Japanese_spaniel/
    └── ...                      # 120+ breed folders
        ├── n02085620_10074      # XML annotation files
        └── ...
images/
├── n02085620-Chihuahua/         # Corresponding image folders
├── n02085782-Japanese_spaniel/
└── ...
```

### Crawford Cats Structure

```
CAT_00/                          # Category folders
├── 00000001_000.jpg            # Image files
├── 00000001_000.jpg.cat        # Corresponding landmark files
├── 00000001_005.jpg
├── 00000001_005.jpg.cat
└── ...
CAT_01/
├── 00000002_000.jpg
├── 00000002_000.jpg.cat
└── ...
```

### Borhanitrash Cats Structure

```
cats/                            # Single folder with cat images
├── cat1.jpg
├── cat2.jpg
├── cat3.jpg
└── ...
```

## 🔧 Customization Options

### Adjust Sample Sizes

```python
# For quick testing
features = loader.load_dogs_cats_images(max_samples_per_class=50)

# For research
features = loader.load_dogs_cats_images(max_samples_per_class=2000)

# For production
features = loader.load_combined_dataset(max_samples_per_dataset=1000)
```

### Control Dataset Selection

```python
# Load only specific datasets
try:
    features1, species1, ages1 = loader.load_stanford_dogs(max_breeds=10)
    features2, species2, ages2 = loader.load_crawford_cats(max_categories=5)

    # Combine manually
    all_features = np.vstack([features1, features2])
    all_species = species1 + species2
    all_ages = ages1 + ages2
except Exception as e:
    print(f"Dataset loading failed: {e}")
```

### Feature Extraction Customization

```python
# The loader automatically handles:
# - Face detection failures → Synthetic features
# - Missing landmarks → Image-based features
# - Corrupted files → Skip and continue
# - Memory constraints → Sample limits
```

## 📊 Data Quality and Validation

### Automatic Validation

- **File Integrity**: Checks if images can be loaded
- **Feature Consistency**: Ensures 8-dimensional feature vectors
- **Label Consistency**: Validates species and age labels
- **Error Handling**: Graceful fallbacks for failed loads

### Quality Metrics

- **Success Rate**: Percentage of successfully processed images
- **Feature Distribution**: Statistical validation of extracted features
- **Species Balance**: Ensures balanced cat/dog representation
- **Memory Usage**: Efficient loading with configurable limits

## 🚨 Common Issues and Solutions

### Issue: Face Detection Fails

**Solution**: The loader automatically falls back to synthetic features

```python
# Check if face detection is working
features = loader.load_dogs_cats_images(max_samples_per_class=10)
# If many synthetic features, consider adjusting face detection parameters
```

### Issue: Memory Errors

**Solution**: Reduce sample sizes

```python
# Start small and scale up
features = loader.load_combined_dataset(max_samples_per_dataset=100)
# If successful, increase gradually
features = loader.load_combined_dataset(max_samples_per_dataset=500)
```

### Issue: Slow Loading

**Solution**: Use smaller sample sizes for development

```python
# Development/testing
features = loader.load_dogs_cats_images(max_samples_per_class=50)

# Production
features = loader.load_dogs_cats_images(max_samples_per_class=1000)
```

## 🔬 Advanced Usage

### Integration with OCI Calculator

```python
from cats_vs_dogs_cuteness import OCICalculator, CutenessAnalyzer

# Load data
features, species, ages = loader.load_combined_dataset(max_samples_per_dataset=500)

# Train OCI model
oci_calculator = OCICalculator(method="lda")
oci_calculator.train(features, ages)

# Calculate cuteness scores
oci_scores = oci_calculator.predict_oci(features)

# Analyze results
analyzer = CutenessAnalyzer()
species_results = analyzer.analyze_species(oci_scores, species)
```

### Custom Feature Extraction

```python
# Extend the loader for custom features
class CustomKaggleLoader(KaggleDatasetLoader):
    def _extract_custom_features(self, img_path):
        # Your custom feature extraction logic
        pass
```

## 📈 Performance Considerations

### Memory Usage

- **Small datasets**: 100-500 samples per class (~50-250MB)
- **Medium datasets**: 500-1000 samples per class (~250MB-1GB)
- **Large datasets**: 1000+ samples per class (1GB+)

### Processing Time

- **Image loading**: ~0.1-0.5 seconds per image
- **Face detection**: ~0.2-1.0 seconds per image
- **Feature extraction**: ~0.1-0.3 seconds per image
- **Total**: ~0.5-2.0 seconds per image

### Optimization Tips

1. **Start small**: Begin with 50-100 samples for testing
2. **Batch processing**: Process datasets in chunks
3. **Parallel processing**: Use multiple processes for large datasets
4. **Caching**: Save extracted features to avoid reprocessing

## 🎯 Best Practices

### 1. **Dataset Selection**

- **Quick testing**: Use dogs-cats-images (balanced, simple)
- **Research**: Use Stanford dogs + Crawford cats (detailed annotations)
- **Production**: Use combined dataset (comprehensive coverage)

### 2. **Sample Sizing**

- **Development**: 50-100 samples per class
- **Validation**: 200-500 samples per class
- **Research**: 500-1000 samples per class
- **Production**: 1000+ samples per class

### 3. **Error Handling**

```python
try:
    features, species, ages = loader.load_combined_dataset(
        max_samples_per_dataset=500
    )
except Exception as e:
    print(f"Loading failed: {e}")
    # Fallback to synthetic data or smaller samples
    features, species, ages = loader.load_combined_dataset(
        max_samples_per_dataset=100
    )
```

### 4. **Data Validation**

```python
# Always validate loaded data
if len(features) > 0:
    print(f"✅ Data loaded successfully: {len(features)} samples")
    print(f"Feature dimensions: {features.shape}")
    print(f"Species distribution: {dict(Counter(species))}")
else:
    print("❌ No data loaded")
```

## 🚀 Next Steps

1. **Explore Datasets**: Run `test_kaggle_loader.py` to verify functionality
2. **Run Examples**: Execute `kaggle_experiment.py` for comprehensive analysis
3. **Customize**: Modify feature extraction for your specific needs
4. **Scale Up**: Increase sample sizes for production use
5. **Integrate**: Combine with your existing analysis pipeline

## 📚 Additional Resources

- **Package Documentation**: See `docs/` folder for detailed implementation
- **Example Scripts**: `kaggle_example.py` and `test_kaggle_loader.py`
- **Core Classes**: `OCICalculator`, `CutenessAnalyzer` for analysis
- **CLI Interface**: `cats-vs-dogs-cuteness` command for experiments

---

**Happy dataset exploration! 🎉**

The Kaggle datasets provide a rich foundation for objective cuteness analysis, combining the best of image processing, facial landmarks, and breed annotations for comprehensive scientific research.
