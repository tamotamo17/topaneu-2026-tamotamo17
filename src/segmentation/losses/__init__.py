"""Segmentation Losses.

Available losses:
- DiceLoss: Soft Dice loss for segmentation
- BCEDiceLoss: Combined BCE and Dice loss
"""

from src.segmentation.losses.loss import BCEDiceLoss, DiceLoss

__all__ = ["DiceLoss", "BCEDiceLoss"]
