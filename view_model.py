from tensorflow.keras.models import load_model

# Load the .h5 model
model = load_model("model.h5")

# Display the model summary
model.summary()