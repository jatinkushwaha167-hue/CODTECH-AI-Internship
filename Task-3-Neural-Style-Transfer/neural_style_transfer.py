import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load and preprocess image
def load_image(image_path, max_dim=512):
    img = Image.open(image_path)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img)

# Convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor.numpy()
    tensor = tensor[0]

    # VGG19 mean values
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.68

    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)


# Load images
content_image = load_image("content.jpg")
style_image = load_image("style.jpg")

# Load VGG19 model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Select layers
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Build model
def vgg_layers(layer_names):
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

style_extractor = vgg_layers(style_layers)
content_extractor = vgg_layers(content_layers)

# Extract features
style_targets = style_extractor(style_image)
content_targets = content_extractor(content_image)

# Initialize generated image
generated_image = tf.Variable(content_image, dtype=tf.float32)

# Gram matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# Optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.02)

# Loss weights
style_weight = 1e-2
content_weight = 1e4

# Training step
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        style_outputs = style_extractor(image)
        content_outputs = content_extractor(image)

        style_loss = tf.add_n([
            tf.reduce_mean((gram_matrix(style_outputs[i]) -
                             gram_matrix(style_targets[i])) ** 2)
            for i in range(num_style_layers)
        ])

        content_loss = tf.add_n([
            tf.reduce_mean((content_outputs[i] -
                             content_targets[i]) ** 2)
            for i in range(num_content_layers)
        ])

        total_loss = style_weight * style_loss + content_weight * content_loss

    grad = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, -103.939, 151.061))

# Run style transfer
epochs = 300
for i in range(epochs):
    train_step(generated_image)
    if i % 50 == 0:
        print(f"Epoch {i} completed")

# Save output
final_image = tensor_to_image(generated_image)

final_image.save("styled_output.png")
print("✅ styled_output.png saved successfully")




