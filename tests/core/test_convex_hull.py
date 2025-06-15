import pytest
import numpy as np
from philosophical_analysis.core.convex_hull import ConvexHullClassifier

class TestConvexHullClassifier:
    """Unit tests for the ConvexHullClassifier."""

    def setup_method(self):
        """Set up a classifier and synthetic data for testing."""
        self.features = ['feature1', 'feature2']
        self.classifier = ConvexHullClassifier(features=self.features)
        
        # Synthetic data: class 1 (coherent) forms a clear cluster
        self.X_train = np.array([
            [1, 1], [1, 2], [2, 1], [2, 2],  # Class 1 (inside the hull)
            [5, 5], [5, 6], [6, 5], [6, 6]   # Class 0 (incoherent)
        ])
        self.y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    def test_initialization(self):
        """Test that the classifier initializes correctly."""
        assert self.classifier.features == self.features
        assert not self.classifier.is_fitted

    def test_fit(self):
        """Test the fit method to ensure hulls are created."""
        self.classifier.fit(self.X_train, self.y_train)
        assert self.classifier.is_fitted
        assert 1 in self.classifier.hulls  # Hull for class 1
        assert 0 in self.classifier.hulls  # Hull for class 0
        assert self.classifier.hulls[1] is not None

    def test_predict_single_point_inside_hull(self):
        """Test prediction for a point clearly inside a class hull."""
        self.classifier.fit(self.X_train, self.y_train)
        point_inside = np.array([1.5, 1.5])
        prediction, confidence = self.classifier.predict_single(point_inside)
        
        assert prediction == 1
        assert confidence > 0.5

    def test_predict_single_point_outside_hulls(self):
        """Test prediction for a point outside of any class hull."""
        self.classifier.fit(self.X_train, self.y_train)
        point_outside = np.array([3.5, 3.5]) # Equidistant from both clusters
        prediction, confidence = self.classifier.predict_single(point_outside)

        # The predicted class can be either 0 or 1 depending on minor distance differences.
        # The key is that the confidence should be very low (close to 0.5).
        assert prediction in [0, 1]
        assert 0.4 < confidence < 0.6

    def test_cross_validate(self):
        """Test the leave-one-out cross-validation."""
        # With this clean data, accuracy should be perfect
        results = self.classifier.cross_validate(self.X_train, self.y_train)
        assert results['accuracy'] == 1.0
        # F1 score might not be perfect due to the nature of LOOCV with small samples
        assert results['f1_score'] >= 0.8

    def test_fit_with_insufficient_data(self):
        """Test that fitting with too few points for a class hull raises a warning."""
        X_bad = np.array([[1,1], [1,2], [5,5]]) # Only 2 points for class 1
        y_bad = np.array([1, 1, 0])
        
        with pytest.warns(UserWarning, match="Insufficient points for 2D convex hull"):
            self.classifier.fit(X_bad, y_bad)
        
        # The hull for class 1 should not have been created
        assert self.classifier.hulls.get(1) is None
        # The hull for class 0 should also not be created as it has only one point
        assert self.classifier.hulls.get(0) is None
