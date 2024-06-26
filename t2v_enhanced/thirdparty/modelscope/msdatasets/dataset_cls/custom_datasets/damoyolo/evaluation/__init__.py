# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright © Alibaba, Inc. and its affiliates.

from .. import datasets
from .coco import coco_evaluation


def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        **kwargs)
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError(
            'Unsupported dataset type {}.'.format(dataset_name))
