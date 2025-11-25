#!/usr/bin/env python
"""
IndoorLoc Quick Start Example

This example demonstrates basic usage of the IndoorLoc framework.
"""
import numpy as np
import indoorloc as iloc


def main():
    print("=" * 60)
    print("IndoorLoc Quick Start Example")
    print("=" * 60)

    # =========================================================
    # 1. Create synthetic training data
    # =========================================================
    print("\n1. Creating synthetic training data...")

    np.random.seed(42)
    num_train = 500
    num_test = 100
    num_aps = 520  # UJIndoorLoc has 520 APs

    # Generate random WiFi fingerprints
    # In real scenario, these would be RSSI values from WiFi scans
    def generate_signal():
        rssi = np.random.uniform(-100, -30, num_aps).astype(np.float32)
        # Set some APs as not detected (100 in UJIndoorLoc)
        not_detected = np.random.choice(num_aps, size=int(num_aps * 0.8), replace=False)
        rssi[not_detected] = 100
        return iloc.WiFiSignal(rssi_values=rssi)

    def generate_location():
        return iloc.Location.from_coordinates(
            x=np.random.uniform(0, 300),
            y=np.random.uniform(0, 300),
            floor=np.random.randint(0, 5),
            building_id=str(np.random.randint(0, 3))
        )

    train_signals = [generate_signal() for _ in range(num_train)]
    train_locations = [generate_location() for _ in range(num_train)]

    test_signals = [generate_signal() for _ in range(num_test)]
    test_locations = [generate_location() for _ in range(num_test)]

    print(f"   Training samples: {num_train}")
    print(f"   Test samples: {num_test}")
    print(f"   Number of APs: {num_aps}")

    # =========================================================
    # 2. Create and train a k-NN localizer
    # =========================================================
    print("\n2. Creating and training k-NN localizer...")

    # Method 1: Direct creation
    model = iloc.create_model('KNNLocalizer', k=5, weights='distance')
    print(f"   Created model: {model}")

    # Train the model
    model.fit(train_signals, train_locations)
    print(f"   Model trained: {model.is_trained}")

    # =========================================================
    # 3. Make predictions
    # =========================================================
    print("\n3. Making predictions...")

    # Single prediction
    result = model.predict(test_signals[0])
    print(f"   Single prediction:")
    print(f"     Position: ({result.x:.2f}, {result.y:.2f})")
    print(f"     Floor: {result.floor}")
    print(f"     Building: {result.building}")
    print(f"     Confidence: {result.confidence:.3f}")
    print(f"     Uncertainty: {result.location.position_uncertainty:.2f}m")

    # Timed prediction
    result_timed = model.predict_timed(test_signals[0])
    print(f"     Latency: {result_timed.latency_ms:.2f}ms")

    # =========================================================
    # 4. Batch prediction
    # =========================================================
    print("\n4. Batch prediction...")

    results = model.predict_batch(test_signals)
    print(f"   Predicted {len(results)} locations")

    # Calculate simple error statistics
    position_errors = []
    floor_correct = 0
    building_correct = 0

    for pred, gt in zip(results, test_locations):
        error = pred.location.distance_to(gt)
        position_errors.append(error)

        if pred.floor == gt.floor:
            floor_correct += 1
        if pred.building == gt.building_id:
            building_correct += 1

    print(f"\n   Evaluation Results:")
    print(f"     Mean Position Error: {np.mean(position_errors):.2f}m")
    print(f"     Median Position Error: {np.median(position_errors):.2f}m")
    print(f"     75th Percentile Error: {np.percentile(position_errors, 75):.2f}m")
    print(f"     Floor Accuracy: {floor_correct / len(results) * 100:.1f}%")
    print(f"     Building Accuracy: {building_correct / len(results) * 100:.1f}%")

    # =========================================================
    # 5. Save and load model
    # =========================================================
    print("\n5. Saving and loading model...")

    model.save('knn_model.pkl')
    print("   Model saved to knn_model.pkl")

    loaded_model = iloc.create_model('KNNLocalizer')
    loaded_model.load('knn_model.pkl')
    print(f"   Model loaded: {loaded_model.is_trained}")

    # Verify loaded model works
    result_loaded = loaded_model.predict(test_signals[0])
    print(f"   Loaded model prediction: ({result_loaded.x:.2f}, {result_loaded.y:.2f})")

    # =========================================================
    # 6. List available models
    # =========================================================
    print("\n6. Available models:")
    for model_name in iloc.list_models():
        print(f"   - {model_name}")

    print("\n" + "=" * 60)
    print("Quick Start completed successfully!")
    print("=" * 60)

    # Cleanup
    import os
    if os.path.exists('knn_model.pkl'):
        os.remove('knn_model.pkl')


if __name__ == '__main__':
    main()
