import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from preprocessing.data_preprocessor import DataPreprocessor
from models.bayesian_nn import BayesianNN
from models.trainer import Trainer

# Load data
def load_data():
    with open("data/sample_data.txt", "r") as file:
        text_data = file.readlines()
    return text_data

# Load initial data
text_data = load_data()

# Initialize the DataPreprocessor
preprocessor = DataPreprocessor(text_data)

# Generate training samples
X, y, identified_entities = preprocessor.generate_training_samples()

print("Preprocessed Data")
print("Tokens and Labels:")
for feature, label in zip(X, y):
    print(f"Token: {feature['token']}, Token ID: {feature['token_id']}, x: {feature['x']}, y: {feature['y']}, Label: {label}")

print("\nIdentified Entities:")
for entity, label in identified_entities:
    print(f"Entity: {entity}, Label: {label}")

# Encode tokens and labels
token_ids = [feature['token_id'] for feature in X]
positions = [feature['x'] for feature in X]

label_encoder = LabelEncoder()
label_encoder.fit(y)
encoded_labels = label_encoder.transform(y)
encoded_labels = to_categorical(encoded_labels)

sequence_length = max(positions) + 1
X_padded = pad_sequences([token_ids], maxlen=sequence_length, padding='post')
y_padded = pad_sequences([encoded_labels], maxlen=sequence_length, padding='post')

train_data = X_padded
train_labels = y_padded

# Hyperparameters
vocab_size = len(preprocessor.tokenizer.vocab)
embedding_dim = 768  # BERT base model embedding size
output_dim = len(label_encoder.classes_)
num_bayesian_layers = 3
units = 64

# Instantiate the model
model = BayesianNN(vocab_size, embedding_dim, output_dim, num_bayesian_layers, units)

# Instantiate the trainer
trainer = Trainer(model, train_data, train_labels, batch_size=32, epochs=10, learning_rate=0.001)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
trainer.save_model("bayesian_nn_model.h5")

# Load new data with potentially new entity categories
new_text_data = [
    "Introducing a new entity type {{New Entity}}{{NEW_ENTITY_TYPE}} in the dataset."
]

# Update preprocessor and generate new samples
new_preprocessor = DataPreprocessor(new_text_data)
new_X, new_y, new_identified_entities = new_preprocessor.generate_training_samples()

print("\nNew Preprocessed Data")
print("Tokens and Labels:")
for feature, label in zip(new_X, new_y):
    print(f"Token: {feature['token']}, Token ID: {feature['token_id']}, x: {feature['x']}, y: {feature['y']}, Label: {label}")

print("\nNew Identified Entities:")
for entity, label in new_identified_entities:
    print(f"Entity: {entity}, Label: {label}")

# Check for new entity categories
new_tokens = [feature['token_id'] for feature in new_X]
new_positions = [feature['x'] for feature in new_X]

# Fit and transform new labels
new_label_encoder = LabelEncoder()
new_label_encoder.fit(new_y)
new_encoded_labels = new_label_encoder.transform(new_y)
new_encoded_labels = to_categorical(new_encoded_labels)

# Check for new categories
new_classes = set(new_label_encoder.classes_)
existing_classes = set(label_encoder.classes_)

if new_classes != existing_classes:
    print("New entity categories detected.")
    combined_classes = sorted(new_classes.union(existing_classes))
    combined_output_dim = len(combined_classes)
    
    # Update label encoders and transform labels
    combined_label_encoder = LabelEncoder()
    combined_label_encoder.fit(combined_classes)
    
    new_encoded_labels = combined_label_encoder.transform(new_y)
    new_encoded_labels = to_categorical(new_encoded_labels, num_classes=combined_output_dim)
    
    # Incremental training
    new_X_padded = pad_sequences([new_tokens], maxlen=sequence_length, padding='post')
    trainer.incremental_train(new_X_padded, new_encoded_labels, combined_output_dim, epochs=5)
    trainer.evaluate()

# Save the updated model
trainer.save_model("bayesian_nn_model_updated.h5")

