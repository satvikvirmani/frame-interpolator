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
    batch_size=1
)
train_ds, val_ds = dataset.get_datasets()

unet = UNet(img_height=256, img_width=448)

trainer = Trainer(
    model=unet.get_model(),
    optimizer=optimizer_with_cosine_annealing,
    loss_fn=combined_loss
)
trainer.train_model(train_ds, val_ds, epochs=50)