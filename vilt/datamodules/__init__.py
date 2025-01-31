from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .memes_datamodule import MemesDataModule
from .memes_classification_datamodule import MemesClassificationDataModule

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "memes": MemesDataModule,
    "classification memes": MemesClassificationDataModule
}
