# from flask import Flask, jsonify, request
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from waitress import serve

# app = Flask(__name__)

# class_names = ['acne', 'atopic_dermatitis', 'normal_skin', 'psoriasis', 'scabies', 'warts']
# class_titles = {'acne': 'Acne', 'atopic_dermatitis': 'Atopic Dermatitis', 'normal_skin': 'Normal Skin', 'psoriasis': 'Psoriasis', 'scabies': 'Scabies', 'warts': 'Warts'}
# model = tf.keras.models.load_model('model\FinalModelFinetunedV2.h5')

# def preprocess_image(image):
#     image = tf.image.resize(image, (224, 224))
#     image = tf.keras.applications.resnet50.preprocess_input(image)
#     return image

# @app.route('/upload', methods=['POST', 'GET'])
# def predict():
#     file = request.files['image']
#     image_path = "./images/" + file.filename
#     file.save(image_path)
#     image = Image.open(file)
#     image = np.array(image)
#     image = preprocess_image(image)

#     # Prediction
#     class_probs = model.predict(np.array([image]))[0]
#     class_index = np.argmax(class_probs)
#     class_preds = round(float(np.max(class_probs)*100), 2)

#     class_label = class_names[class_index]
#     class_title = class_titles[class_label]
#     print(class_label)
#     print(class_probs)
#     print(class_preds)

#     # Return dictionary with class_label and class_preds
#     results = {'class_title': class_title, 'class_preds': class_preds}
#     return jsonify(results)
#     # return 'Hello World!'
    

# if __name__ == '__main__':
#     serve(app, host='192.168.205.117', port=5000, threads = 10)

from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
from PIL import Image
from waitress import serve

app = Flask(__name__)

class_names = ['acne', 'atopic_dermatitis', 'normal_skin', 'psoriasis', 'scabies', 'warts']
class_titles = {'acne': 'Acne', 'atopic_dermatitis': 'Atopic Dermatitis', 'normal_skin': 'Normal Skin', 'psoriasis': 'Psoriasis', 'scabies': 'Scabies', 'warts': 'Warts'}

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model\optimized_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the expected input shape of the model
interpreter.resize_tensor_input(input_details[0]['index'], (1, 224, 224, 3))
interpreter.allocate_tensors()

def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


@app.route('/upload', methods=['POST', 'GET'])
def predict():
    file = request.files['image']
    image_path = "./images/" + file.filename
    file.save(image_path)
    image = Image.open(file)
    preprocessed_image = preprocess_image(image)

    # Prediction
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    class_probs = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(class_probs[0])
    class_preds = round(float(class_probs[0][class_index] * 100), 2)

    class_label = class_names[class_index]
    class_title = class_titles[class_label]
    print(class_label)
    print(class_probs)
    print(class_preds)

    # Return dictionary with class_label and class_preds
    results = {'class_title': class_title, 'class_preds': class_preds}
    return jsonify(results)
    # return 'Hello World!'
    

if __name__ == '__main__':
    serve(app, host='192.168.205.117', port=5000, threads = 10)
