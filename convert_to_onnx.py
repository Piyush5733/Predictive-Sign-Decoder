import tensorflow as tf
import tf2onnx
import onnx
import os

def convert():
    model_path = "dynamic_sign_model.h5"
    output_path = "dynamic_sign_model.onnx"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print("Loading Keras model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Converting to ONNX...")
    # opset 13 is usually safe for modern runtimes
    spec = (tf.TensorSpec((None, 30, 126), tf.float32, name="input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=spec, 
        opset=13, 
        output_path=output_path
    )
    
    print(f"Success! Model saved to {output_path}")

if __name__ == "__main__":
    convert()
