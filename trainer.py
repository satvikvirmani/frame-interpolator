import tensorflow as tf
from tqdm.notebook import tqdm
import os

class Trainer:
    def __init__(self, model, optimizer, loss_fn, checkpoint_dir='checkpoints/'):
        self.model = model
        self.optimizer_fn = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.ckpt = None
        self.ckpt_manager = None

    @tf.function
    def train_step(self, input_frames, target_frame):
        with tf.GradientTape() as tape:
            predicted_frame = self.model(input_frames, training=True)
            loss, l1, ssim, perceptual, edge = self.loss_fn(target_frame, predicted_frame)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, l1, ssim, perceptual, edge
    
    @tf.function
    def test_step(self, input_frames, target_frame):
        predicted_frame = self.model(input_frames, training=False)
        total_loss, l1, ssim, perceptual, edge = self.loss_fn(target_frame, predicted_frame)
        return total_loss, l1, ssim, perceptual, edge
    
    def set_optimizer(self, decay_steps):
        self.optimizer = self.optimizer_fn(decay_steps, INITIAL_LEARNING_RATE=1e-4)
    
    def set_checkpoint_manager(self):
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimizer,
            model=self.model
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_dir, max_to_keep=2
        )
    
    def set_dataset(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.set_optimizer(train_ds)
        self.set_checkpoint_manager()

    def train_model(self, epochs):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Restored from {self.ckpt_manager.latest_checkpoint}")
        else:
            print("No checkpoint found. Starting fresh training.")

        best_test_loss = float('inf')
        patience = 10
        patience_counter = 0

        start_epoch = int(self.ckpt.step.numpy())

        for epoch in range(start_epoch, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_metrics = {
                'total': tf.keras.metrics.Mean(),
                'l1': tf.keras.metrics.Mean(),
                'ssim': tf.keras.metrics.Mean(),
                'perceptual': tf.keras.metrics.Mean(),
                'edge': tf.keras.metrics.Mean()
            }

            prog_bar = tqdm(self.train_ds, desc=f"Training", unit="batch")
            for inputs, targets in prog_bar:
                total, l1, ssim, perceptual, edge = self.train_step(inputs, targets)
                train_metrics['total'].update_state(total)
                train_metrics['l1'].update_state(l1)
                train_metrics['ssim'].update_state(ssim)
                train_metrics['perceptual'].update_state(perceptual)
                train_metrics['edge'].update_state(edge)

                prog_bar.set_postfix({
                    'loss': f"{train_metrics['total'].result():.4f}",
                    'lr': f"{self.optimizer.learning_rate.numpy():.6f}"
                })

            print(f"\nTrain - Total: {train_metrics['total'].result():.4f}, "
                f"L1: {train_metrics['l1'].result():.4f}, "
                f"SSIM: {train_metrics['ssim'].result():.4f}, "
                f"Perceptual: {train_metrics['perceptual'].result():.4f}, "
                f"Edge: {train_metrics['edge'].result():.4f}")

            self.ckpt.step.assign_add(1)
            save_path = self.ckpt_manager.save()
            print(f"Saved checkpoint for epoch {epoch}: {save_path}")

            test_metrics = self.test_model(training=True)

            current_test_loss = test_metrics['total'].result()
            if current_test_loss < best_test_loss:
                best_test_loss = current_test_loss
                patience_counter = 0
                print(f"New best test loss: {best_test_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")

                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

        print("\nTraining complete!")

    def test_model(self, training=False):
        if(training):
            if self.ckpt_manager.latest_checkpoint:
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print(f"Loaded from {self.ckpt_manager.latest_checkpoint}")
            else:
                print("No checkpoint found.")

        test_metrics = {
            'total': tf.keras.metrics.Mean(),
            'l1': tf.keras.metrics.Mean(),
            'ssim': tf.keras.metrics.Mean(),
            'perceptual': tf.keras.metrics.Mean(),
            'edge': tf.keras.metrics.Mean()
        }

        for inputs, targets in tqdm(self.test_ds, desc="Testing", unit="batch"):
            total, l1, ssim, perceptual, edge = self.test_step(inputs, targets)
            test_metrics['total'].update_state(total)
            test_metrics['l1'].update_state(l1)
            test_metrics['ssim'].update_state(ssim)
            test_metrics['perceptual'].update_state(perceptual)
            test_metrics['edge'].update_state(edge)

        print(f"Test - Total: {test_metrics['total'].result():.4f}, "
            f"L1: {test_metrics['l1'].result():.4f}, "
            f"SSIM: {test_metrics['ssim'].result():.4f}, "
            f"Perceptual: {test_metrics['perceptual'].result():.4f}, "
            f"Edge: {test_metrics['edge'].result():.4f}")

        return test_metrics

    def visualize(self, save_dir=None):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            print(f"Restored from {self.ckpt_manager.latest_checkpoint}")
        else:
            print("No checkpoint found. Visualizing with random weights.")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print("Visualizing 5 random samples...")
                
        i = 0
        for inputs, targets in tqdm(self.test_ds.take(1), desc="Visualizing", unit="batch"):
            predictions = self.model(inputs, training=False)
            
            print(f"\nBatch stats:")
            print(f"Input range: [{tf.reduce_min(inputs):.3f}, {tf.reduce_max(inputs):.3f}]")
            print(f"Target range: [{tf.reduce_min(targets):.3f}, {tf.reduce_max(targets):.3f}]")
            print(f"Pred range: [{tf.reduce_min(predictions):.3f}, {tf.reduce_max(predictions):.3f}]")
            print(f"Pred mean: {tf.reduce_mean(predictions):.3f}")

            for j in range(inputs.shape[0]):
                frame0 = inputs[j, ..., :3]
                frame2 = inputs[j, ..., 3:]
                target = targets[j]
                pred = predictions[j]
                
                if save_dir:
                    def save_image(img, filename):
                        img = tf.clip_by_value(img, 0.0, 1.0)
                        img_uint8 = tf.image.convert_image_dtype(img, tf.uint8)
                        encoded = tf.io.encode_jpeg(img_uint8)
                        tf.io.write_file(os.path.join(save_dir, filename), encoded)
                    
                    save_image(frame0, f"sample_{i}_input_0.jpg")
                    save_image(frame2, f"sample_{i}_input_1.jpg")
                    save_image(target, f"sample_{i}_target.jpg")
                    save_image(pred, f"sample_{i}_prediction.jpg")
                
                i += 1