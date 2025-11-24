from models.unet import UNet
from models.loss import combined_loss
from models.optimiser import optimizer_with_cosine_annealing

from trainer import Trainer
from data.dataset import Dataset

dataset = Dataset(
    video_dir='data/videos',
    output_dir='data/dataset',
    resize=(448, 256),
    frame_skip=4,
    group_stride=2,
    max_frames=1000,
    test_ratio=0.2,
    img_quality=90,
    batch_size=16,
    skip_processing=True
)
train_ds, test_ds = dataset.get_datasets()

unet = UNet(img_height=256, img_width=448)

trainer = Trainer(
    model=unet.get_model(),
    optimizer=optimizer_with_cosine_annealing,
    loss_fn=combined_loss
)
trainer.set_dataset(train_ds, test_ds)
# trainer.train_model(epochs=50)
# trainer.test_model()
trainer.visualize(save_dir='data/visualize')