# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.text_driven_segmentation,
    module_name=Pipelines.text_driven_segmentation)
class TextDrivenSegmentationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
            model: model id on modelscope hub.
        """
        super().__init__(model=model, auto_collate=False, **kwargs)

    def preprocess(self, input: Dict) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input['image'])
        img_tensor, ori_h, ori_w, crop_h, crop_w = self.model.preprocess(img)
        result = {
            'img': img_tensor,
            'ori_h': ori_h,
            'ori_w': ori_w,
            'crop_h': crop_h,
            'crop_w': crop_w,
            'text': input['text'],
        }
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        outputs = self.model.inference(input['img'], input['text'])
        result = {
            'data': outputs,
            'ori_h': input['ori_h'],
            'ori_w': input['ori_w'],
            'crop_h': input['crop_h'],
            'crop_w': input['crop_w'],
        }
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = self.model.postprocess(inputs['data'], inputs['crop_h'],
                                      inputs['crop_w'], inputs['ori_h'],
                                      inputs['ori_w'])
        outputs = {OutputKeys.MASKS: data}
        return outputs
