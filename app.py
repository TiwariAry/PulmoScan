from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model once when app starts
model = tf.keras.models.load_model('model.h5')
last_conv_layer_name = 'out_relu'  # MobileNetV2 last conv layer

def generate_gradcam(img_path, model, last_conv_layer_name):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Build a model that maps input -> (conv activations, predictions)
    grad_model = tf.keras.models.Model(
        inputs=model.input,  # prefer .input in Keras 3 to avoid structure warnings
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Unwrap predictions if model.output is a list/tuple
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]  # shape: (1, num_classes)

        # Pick the top predicted class for this image
        class_index_tensor = tf.argmax(predictions[0])  # scalar tensor
        # Build scalar loss for Grad-CAM over the chosen class
        loss = tf.gather(predictions, class_index_tensor, axis=1)  # shape: (1,)

    # Gradients of loss w.r.t. conv maps
    grads = tape.gradient(loss, conv_outputs)  # shape: (1, H, W, C)

    # Channel-wise global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # Weight conv outputs by pooled grads
    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=([2], [0]))  # (H, W)

    # Normalize heatmap safely
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap)
    heatmap = heatmap / (denom + 1e-8)
    heatmap = heatmap.numpy()

    # Load original image (OpenCV is BGR)
    img_original = cv2.imread(img_path)
    img_original = cv2.resize(img_original, (224, 224))
    img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    # Create color heatmap
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(img_original_rgb, 0.6, heatmap_color_rgb, 0.4, 0)

    # Confidence for the chosen class
    class_index = int(class_index_tensor.numpy())
    confidence = float(predictions.numpy()[0, class_index])

    return img_original_rgb, heatmap_color, overlay, confidence


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