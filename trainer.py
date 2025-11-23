import tensorflow as tf
from tqdm import tqdm
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
    def val_step(self, input_frames, target_frame):
        predicted_frame = self.model(input_frames, training=False)
        total_loss, l1, ssim, perceptual, edge = self.loss_fn(target_frame, predicted_frame)
        return total_loss, l1, ssim, perceptual, edge
    
    def train_model(self, train_ds, val_ds, epochs):
        self.optimizer = self.optimizer_fn(train_ds, INITIAL_LEARNING_RATE=1e-4)

        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimizer,
            model=self.model
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_dir, max_to_keep=2
        )

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Restored from {self.ckpt_manager.latest_checkpoint}")
        else:
            print("No checkpoint found. Starting fresh training.")

        best_val_loss = float('inf')
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

            prog_bar = tqdm(train_ds, desc=f"Training", unit="batch")
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

            val_metrics = {
                'total': tf.keras.metrics.Mean(),
                'l1': tf.keras.metrics.Mean(),
                'ssim': tf.keras.metrics.Mean(),
                'perceptual': tf.keras.metrics.Mean(),
                'edge': tf.keras.metrics.Mean()
            }

            for inputs, targets in tqdm(val_ds, desc="Validation", unit="batch"):
                total, l1, ssim, perceptual, edge = self.val_step(inputs, targets)
                val_metrics['total'].update_state(total)
                val_metrics['l1'].update_state(l1)
                val_metrics['ssim'].update_state(ssim)
                val_metrics['perceptual'].update_state(perceptual)
                val_metrics['edge'].update_state(edge)

            print(f"\nTrain - Total: {train_metrics['total'].result():.4f}, "
                f"L1: {train_metrics['l1'].result():.4f}, "
                f"SSIM: {train_metrics['ssim'].result():.4f}, "
                f"Perceptual: {train_metrics['perceptual'].result():.4f}, "
                f"Edge: {train_metrics['edge'].result():.4f}")

            print(f"Val   - Total: {val_metrics['total'].result():.4f}, "
                f"L1: {val_metrics['l1'].result():.4f}, "
                f"SSIM: {val_metrics['ssim'].result():.4f}, "
                f"Perceptual: {val_metrics['perceptual'].result():.4f}, "
                f"Edge: {val_metrics['edge'].result():.4f}")

            self.ckpt.step.assign_add(1)
            save_path = self.ckpt_manager.save()
            print(f"Saved checkpoint for epoch {epoch}: {save_path}")

            current_val_loss = val_metrics['total'].result()
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                print(f"New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")

                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

        print("\nTraining complete!")