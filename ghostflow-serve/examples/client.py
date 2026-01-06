#!/usr/bin/env python3
"""
GhostFlow REST API Client Example

This example demonstrates how to interact with the GhostFlow model serving API.

Usage:
    python client.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:8080"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def main():
    print("🚀 GhostFlow REST API Client Example")
    
    # 1. Health Check
    print_section("1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # 2. Load a Model
    print_section("2. Load Model")
    load_request = {
        "name": "mnist_classifier",
        "path": "/models/mnist.gfcp"
    }
    response = requests.post(f"{BASE_URL}/models/load", json=load_request)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    model_id = result.get("id")
    print(f"\n✓ Model loaded with ID: {model_id}")
    
    # 3. List Models
    print_section("3. List Models")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    models = response.json()
    print(f"Loaded models: {len(models)}")
    for model in models:
        print(f"  - {model['name']} (ID: {model['id']})")
    
    # 4. Get Model Info
    print_section("4. Get Model Info")
    response = requests.get(f"{BASE_URL}/models/{model_id}")
    print(f"Status: {response.status_code}")
    print(f"Model info: {json.dumps(response.json(), indent=2)}")
    
    # 5. Make Predictions
    print_section("5. Make Predictions")
    
    # Create sample input (28x28 image flattened)
    sample_input = [[0.0] * 784]
    
    prediction_request = {
        "inputs": sample_input,
        "shape": [1, 784]
    }
    
    print("Sending prediction request...")
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/models/{model_id}/predict",
        json=prediction_request
    )
    elapsed = (time.time() - start_time) * 1000
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Inference time (server): {result['inference_time_ms']:.2f} ms")
    print(f"Round-trip time: {elapsed:.2f} ms")
    print(f"Output shape: {result['shape']}")
    
    # 6. Batch Predictions
    print_section("6. Batch Predictions")
    
    batch_size = 10
    batch_input = [[0.0] * 784 for _ in range(batch_size)]
    
    batch_request = {
        "inputs": batch_input,
        "shape": [batch_size, 784]
    }
    
    print(f"Sending batch prediction request (batch_size={batch_size})...")
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/models/{model_id}/predict",
        json=batch_request
    )
    elapsed = (time.time() - start_time) * 1000
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Inference time (server): {result['inference_time_ms']:.2f} ms")
    print(f"Round-trip time: {elapsed:.2f} ms")
    print(f"Throughput: {batch_size / (elapsed / 1000):.2f} samples/sec")
    
    # 7. Unload Model
    print_section("7. Unload Model")
    response = requests.post(f"{BASE_URL}/models/{model_id}/unload")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print(f"\n✓ Model unloaded")
    
    # 8. Verify Model Unloaded
    print_section("8. Verify Model Unloaded")
    response = requests.get(f"{BASE_URL}/models")
    models = response.json()
    print(f"Loaded models: {len(models)}")
    
    print("\n" + "="*60)
    print("  ✅ All tests completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to GhostFlow server")
        print("   Make sure the server is running:")
        print("   cargo run --bin ghostflow-serve\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
