#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality of the cuteness analysis system.
Run this script to check if all dependencies are properly installed.
"""

import sys
import os


def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")

    try:
        import cv2

        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False

    try:
        import numpy as np

        print(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False

    try:
        import matplotlib

        print(f"✅ Matplotlib version: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False

    try:
        import seaborn as sns

        print(f"✅ Seaborn version: {sns.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
        return False

    try:
        import sklearn

        print(f"✅ Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False

    try:
        import scipy

        print(f"✅ SciPy version: {scipy.__version__}")
    except ImportError as e:
        print(f"❌ SciPy import failed: {e}")
        return False

    try:
        import tqdm

        print(f"✅ tqdm version: {tqdm.__version__}")
    except ImportError as e:
        print(f"❌ tqdm import failed: {e}")
        return False

    try:
        from PIL import Image

        print(f"✅ Pillow version: {Image.__version__}")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False

    return True


def test_data_directory():
    """Test if the data directory structure exists."""
    print("\n📁 Testing data directory structure...")

    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found")
        return False

    cats_dir = os.path.join(data_dir, "cats")
    dogs_dir = os.path.join(data_dir, "dogs")

    if not os.path.exists(cats_dir):
        print(f"❌ Cats directory '{cats_dir}' not found")
        return False

    if not os.path.exists(dogs_dir):
        print(f"❌ Dogs directory '{dogs_dir}' not found")
        return False

    # Count images
    cat_images = [
        f for f in os.listdir(cats_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    dog_images = [
        f for f in os.listdir(dogs_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"✅ Cats directory: {len(cat_images)} images found")
    print(f"✅ Dogs directory: {len(dog_images)} images found")

    if len(cat_images) == 0:
        print("⚠️  Warning: No cat images found")

    if len(dog_images) == 0:
        print("⚠️  Warning: No dog images found")

    return True


def test_basic_functionality():
    """Test basic functionality of the analysis system."""
    print("\n🧪 Testing basic functionality...")

    try:
        # Test updated analyzer
        from cuteness_analysis import CutenessAnalyzer

        # Create analyzer instance
        analyzer = CutenessAnalyzer()
        print("✅ CutenessAnalyzer created successfully")

        # Test face detector
        detector = analyzer.landmark_detector
        print("✅ Face detector initialized successfully")

        # Test feature extractor
        extractor = analyzer.feature_extractor
        print("✅ Feature extractor initialized successfully")

        # Test OCI calculator
        oci_calc = analyzer.oci_calculator
        print("✅ OCI calculator initialized successfully")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_image_processing():
    """Test basic image processing capabilities."""
    print("\n🖼️  Testing image processing...")

    try:
        import cv2
        import numpy as np

        # Create a simple test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test OpenCV operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        print("✅ Basic image processing operations successful")
        return True

    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False


def test_ml_functionality():
    """Test machine learning functionality."""
    print("\n🤖 Testing machine learning functionality...")

    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        # Test basic ML operations
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)

        predictions = model.predict(X_scaled)
        print("✅ Basic machine learning operations successful")
        return True

    except Exception as e:
        print(f"❌ Machine learning test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Cuteness Analysis System - Installation Test")
    print("=" * 50)

    all_tests_passed = True

    # Test 1: Package imports
    if not test_imports():
        all_tests_passed = False

    # Test 2: Data directory
    if not test_data_directory():
        all_tests_passed = False

    # Test 3: Basic functionality
    if not test_basic_functionality():
        all_tests_passed = False

    # Test 4: Image processing
    if not test_image_processing():
        all_tests_passed = False

    # Test 5: Machine learning
    if not test_ml_functionality():
        all_tests_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! The system is ready to use.")
        print("\n🚀 You can now run:")
        print("   python cuteness_analysis.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Check Python version (3.7+ required)")
        print("   3. Ensure data directory exists with cat/dog images")

    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
