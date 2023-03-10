import argparse
import time
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter
import cv2


def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    if output.shape:
        ordered = np.argpartition(-output, top_k)
        return [(i, output[i]) for i in ordered[:top_k]]
    else:
        return output


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help=r'/home/pi/anomaly/model.tflite', required=True)
    parser.add_argument(
        '--labels', help=r'/home/pi/anomaly/labels.txt', required=True)
    args = parser.parse_args()

    labels = load_labels(args.labels)

    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    rstp_address = 'rtsp://admin:pipipi@192.168.1.22'
    cap = cv2.VideoCapture(rstp_address)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image = Image.open(frame).convert('RGB').resize((width, height), Image.ANTIALIAS)
            start_time = time.time()
            result = classify_image(interpreter, image)
            elapsed_ms = (time.time() - start_time) * 1000
            label_id, prob = result[0]
            print(f'Label: {labels[label_id]}, Probability: {prob}, Elapsed_ms: {elapsed_ms}')
            cap.release()
        else:
            cap.release()


if __name__ == '__main__':
    main()
