
from dataclasses import dataclass
from typing import Optional, Union, Any, List

from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        bboxes = [feature["bbox"] for feature in features]

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_source_length = max(len(bbox) for bbox in bboxes)
        max_target_length = max(len(label) for label in labels)
        if self.pad_to_multiple_of is not None:
            max_source_length = (
                (max_source_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
            max_target_length = (
                (max_target_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        padding_side = self.tokenizer.padding_side

        for feature in features:
            remainder_bbox = [[0, 0, 0, 0]] * (max_source_length - len(feature["bbox"]))
            feature["bbox"] = (
                feature["bbox"] + remainder_bbox if padding_side == "right" else remainder_bbox + feature["bbox"]
            )
            if labels is not None:
                remainder_label = [self.label_pad_token_id] * (max_target_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder_label if padding_side == "right" else remainder_label + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
