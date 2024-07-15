import tensorflow as tf
from sklearn.metrics import classification_report

class Trainer:
    def __init__(self, model, train_data, train_labels, val_data=None, val_labels=None, batch_size=32, epochs=10, learning_rate=0.001):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def train(self):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_data, self.train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        if self.val_data is not None and self.val_labels are not None:
            val_dataset = tf.data.Dataset.from_tensor_slices((self.val_data, self.val_labels)).batch(self.batch_size)
        else:
            val_dataset = None

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self._train_epoch(train_dataset)
            if val_dataset:
                self._validate_epoch(val_dataset)

    def _train_epoch(self, train_dataset):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = self.model(x_batch_train, training=True)
                loss_value = self.loss_fn(y_batch_train, logits)
            
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            if step % 100 == 0:
                print(f"Training loss at step {step}: {loss_value:.4f}")

    def _validate_epoch(self, val_dataset):
        val_loss = 0
        val_steps = 0
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = self.model(x_batch_val, training=False)
            val_loss += self.loss_fn(y_batch_val, val_logits)
            val_steps += 1
        val_loss /= val_steps
        print(f"Validation loss: {val_loss:.4f}")

    def evaluate(self):
        if self.val_data is None or self.val_labels is None:
            print("No validation data provided.")
            return
        val_logits = self.model(self.val_data, training=False)
        val_predictions = tf.argmax(val_logits, axis=1)
        val_true_labels = tf.argmax(self.val_labels, axis=1)
        print(classification_report(val_true_labels, val_predictions))

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)

    def incremental_train(self, new_train_data, new_train_labels, new_output_dim, epochs=5):
        self.model.update_output_layer(new_output_dim)
        self.train_data = new_train_data
        self.train_labels = new_train_labels
        self.epochs = epochs
        self.train()

