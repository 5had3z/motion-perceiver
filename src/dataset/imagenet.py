from dataclasses import dataclass
from typing import List

from konductor.data import DatasetConfig, ModuleInitConfig, Split, DATASET_REGISTRY
from konductor.data.dali import DaliLoaderConfig
from nvidia.dali import fn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.types import DALIDataType, Constant


@dataclass
@DATASET_REGISTRY.register_module("imagenet-1k")
class ImageNet1kCfg(DatasetConfig):
    train_loader: DaliLoaderConfig
    val_loader: DaliLoaderConfig

    crop_size: int = 224
    num_classes: int = 1000

    @property
    def properties(self):
        return self.__dict__

    def get_dataloader(self, split: Split):
        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        pipeline = imagenet_pipeline(cfg=self, split=split, **loader.pipe_kwargs())
        return loader.get_instance(
            pipeline, output_map=["image", "label"], reader_name=split.name.lower()
        )


@pipeline_def()
def imagenet_pipeline(
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    cfg: ImageNet1kCfg,
    split: Split,
    augmentations: List[ModuleInitConfig],
):
    """Basic ImageNet1k Dataloader"""
    image, label = fn.readers.file(
        file_root=str(cfg.basepath / split.name.lower()),
        num_shards=num_shards,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        name=split.name.lower(),
    )

    image = fn.decoders.image(image, device="mixed")
    if split is Split.TRAIN:
        crop_pos_x = fn.random.uniform(range=[0, 1], dtype=DALIDataType.FLOAT)
        crop_pos_y = fn.random.uniform(range=[0, 1], dtype=DALIDataType.FLOAT)
        shorter_size = fn.random.uniform(range=[256, 480], dtype=DALIDataType.FLOAT)
        image = fn.resize_crop_mirror(
            image,
            crop_pos_x=crop_pos_x,
            crop_pos_y=crop_pos_y,
            crop_h=cfg.crop_size,
            crop_w=cfg.crop_size,
            resize_shorter=shorter_size,
            mirror=fn.random.coin_flip(probability=0.5, dtype=DALIDataType.INT32),
        )
    else:
        image = fn.resize_crop_mirror(
            image,
            crop_h=cfg.crop_size,
            crop_w=cfg.crop_size,
            resize_shorter=cfg.crop_size,
        )

    image = fn.cast(fn.transpose(image, perm=[2, 0, 1]), dtype=DALIDataType.FLOAT)
    image = image / Constant(255, dtype=DALIDataType.FLOAT)
    image = fn.normalize(
        image,
        mean=Constant([0.485, 0.456, 0.406], dtype=DALIDataType.FLOAT, shape=[3, 1, 1]),
        stddev=Constant(
            [0.229, 0.224, 0.225], dtype=DALIDataType.FLOAT, shape=[3, 1, 1]
        ),
    )

    label = fn.cast(label.gpu(), dtype=DALIDataType.INT64, device="gpu")
    return image, label
