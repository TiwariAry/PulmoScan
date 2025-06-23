from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model once when app starts
model = tf.keras.models.load_model('model.h5')
last_conv_layer_name = 'Conv_1'  # MobileNetV2 last conv layer

def generate_gradcam(img_path, model, last_conv_layer_name):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Create model that maps input to activations + predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)

    # Load original image with OpenCV and convert for RGB overlay
    img_original = cv2.imread(img_path)
    img_original = cv2.resize(img_original, (224, 224))
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    # Create heatmap color image
    heatmap_resized = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay the heatmap onto the original image
    superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap_color_rgb, 0.4, 0)

    # Return results and prediction confidence
    return img_original, heatmap_color, superimposed_img, predictions.numpy()[0][class_index].item()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = 'uploaded.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('result', filename=filename))
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Generate Grad-CAM images and get confidence
    original, heatmap, overlay, confidence = generate_gradcam(filepath, model, last_conv_layer_name)

    # Save images to disk
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{filename}')
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], f'heatmap_{filename}')
    overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], f'overlay_{filename}')

    cv2.imwrite(original_path, cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if confidence > 0.5:
        label = 'PNEUMONIA'
        confidence *= 100
    else:
        label = 'NORMAL'
        confidence = 100 - confidence * 100

    return render_template('result.html',
                           label=label,
                           confidence=f"{confidence:.2f}%",
                           original_image=url_for('static', filename=f'uploads/original_{filename}'),
                           heatmap_image=url_for('static', filename=f'uploads/heatmap_{filename}'),
                           overlay_image=url_for('static', filename=f'uploads/overlay_{filename}')
                          )

if __name__ == '__main__':
    import os
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)