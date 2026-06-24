"""PyTorch runtime for the N05 word-level CRNN/CTC model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .preprocessing import (
        WordPreprocessSettings,
        load_grayscale_image,
        threshold_for_word_ocr,
    )
except ImportError:
    from preprocessing import (  # type: ignore
        WordPreprocessSettings,
        load_grayscale_image,
        threshold_for_word_ocr,
    )


def prepare_torch_word_tensor(
    crop_path: str | Path,
    settings: WordPreprocessSettings | None = None,
):
    """Prepare one word crop for the PyTorch CRNN checkpoint.

    Args:
        crop_path: Path to the word/text-unit crop.
        settings: Preprocessing settings.

    Returns:
        Tuple of ``(tensor, prepared_shape)``. The tensor has shape
        ``1 x 1 x H x W`` and uses ink as positive evidence.
    """

    import torch

    settings = settings or WordPreprocessSettings(normalize_range="zero_to_one")
    image = load_grayscale_image(crop_path)
    binary = threshold_for_word_ocr(image).astype(np.float32)
    height, width = binary.shape
    if height <= 0 or width <= 0:
        raise ValueError("Word OCR crop has empty dimensions.")

    if settings.dynamic_width:
        scale = settings.target_height / float(height)
        target_width = int(width * scale + settings.padding_px)
        target_width = max(4, target_width + ((4 - target_width) % 4))
        target_height = settings.target_height
        offset_x = (target_width - width * scale) / 2.0
        offset_y = 0.0
    else:
        scale = min(
            settings.target_width / float(width),
            settings.target_height / float(height),
        )
        target_width = settings.target_width
        target_height = settings.target_height
        offset_x = (target_width - width * scale) / 2.0
        offset_y = (target_height - height * scale) / 2.0

    transform = np.float32([[scale, 0, offset_x], [0, scale, offset_y]])
    canvas = np.ones((target_height, target_width), dtype=np.float32) * 255.0
    resized = cv2.warpAffine(
        binary,
        transform,
        dsize=(target_width, target_height),
        dst=canvas,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
    arr = resized / 255.0
    arr = 1.0 - arr
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return tensor, [target_height, target_width]


class WordCRNNRuntime:
    """Lazy wrapper around the trained CRNN/CTC model.

    Args:
        checkpoint_path: Path to the saved ``.pt`` checkpoint.
        device: ``auto``, ``cpu``, or ``cuda``.

    Returns:
        Runtime object capable of greedy CTC inference.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        import torch

        self.torch = torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.token_to_char = {
            int(token): str(char)
            for token, char in checkpoint["token_to_char"].items()
        }
        self.model_name = str(checkpoint.get("model_name", self.checkpoint_path.stem))
        state_dict = checkpoint["model_state_dict"]
        self.length_auxiliary = checkpoint.get("length_auxiliary", {})
        self.bridge_auxiliary = checkpoint.get("bridge_auxiliary", {})
        self.boundary_auxiliary = checkpoint.get("boundary_auxiliary", {})
        self.has_length_head = any(key.startswith("length_head.") for key in state_dict)
        self.has_bridge_head = any(key.startswith("bridge_head.") for key in state_dict)
        self.has_boundary_head = any(key.startswith("boundary_head.") for key in state_dict)
        self.model = _WordCRNN(
            num_tokens=len(self.token_to_char) + 1,
            max_length_class=int(self.length_auxiliary.get("max_length_class", 24)),
            length_auxiliary_enabled=self.has_length_head,
            max_bridge_count_class=int(self.bridge_auxiliary.get("max_bridge_count_class", 24)),
            bridge_auxiliary_enabled=self.has_bridge_head,
            boundary_auxiliary_enabled=self.has_boundary_head,
        ).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def predict(self, crop_path: str | Path, settings: WordPreprocessSettings):
        """Recognize one crop with greedy CTC decoding.

        Args:
            crop_path: Path to the word/text-unit crop.
            settings: Preprocessing settings.

        Returns:
            JSON-safe prediction evidence.
        """

        tensor, prepared_shape = prepare_torch_word_tensor(crop_path, settings)
        tensor = tensor.to(self.device)
        with self.torch.no_grad():
            outputs = self.model(tensor)
            log_probs = outputs["log_probs"]
            probs = log_probs.exp()
            max_probs, tokens = probs.max(dim=2)
        token_sequence = tokens[0].detach().cpu().tolist()
        confidence_sequence = max_probs[0].detach().cpu().tolist()
        decoded, decoded_tokens = _decode_greedy_with_tokens(
            token_sequence,
            confidence_sequence,
            self.token_to_char,
        )
        confidence = float(max_probs[0].detach().cpu().mean().item())
        predicted_length = None
        length_confidence = None
        if outputs["length_logits"] is not None:
            length_probs = outputs["length_logits"].softmax(dim=1)
            length_confidence_tensor, length_prediction_tensor = length_probs.max(dim=1)
            predicted_length = int(length_prediction_tensor[0].detach().cpu().item())
            length_confidence = float(length_confidence_tensor[0].detach().cpu().item())
        predicted_bridge_count = None
        bridge_confidence = None
        if outputs["bridge_logits"] is not None:
            bridge_probs = outputs["bridge_logits"].softmax(dim=1)
            bridge_confidence_tensor, bridge_prediction_tensor = bridge_probs.max(dim=1)
            predicted_bridge_count = int(bridge_prediction_tensor[0].detach().cpu().item())
            bridge_confidence = float(bridge_confidence_tensor[0].detach().cpu().item())
        split_line_candidates = []
        if outputs["boundary_logits"] is not None:
            threshold = float(self.boundary_auxiliary.get("probability_threshold", 0.45))
            boundary_probs = outputs["boundary_logits"].sigmoid()[0].detach().cpu().tolist()
            for step, probability in enumerate(boundary_probs):
                if probability >= threshold:
                    split_line_candidates.append(
                        {
                            "x": int(step * 4),
                            "time_step": int(step),
                            "probability": float(probability),
                        }
                    )
        return {
            "model_name": self.model_name,
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "prepared_shape": prepared_shape,
            "text": decoded,
            "confidence": confidence,
            "decoded_length": len(decoded),
            "tokens": decoded_tokens,
            "predicted_length": predicted_length,
            "length_confidence": length_confidence,
            "predicted_bridge_count": predicted_bridge_count,
            "bridge_confidence": bridge_confidence,
            "split_line_candidates": split_line_candidates,
        }


class _WordCRNN:
    """Internal architecture matching Cyber Lin Kuei word trainer."""

    def __new__(
        cls,
        num_tokens: int,
        max_length_class: int = 24,
        length_auxiliary_enabled: bool = False,
        max_bridge_count_class: int = 24,
        bridge_auxiliary_enabled: bool = False,
        boundary_auxiliary_enabled: bool = False,
    ):
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(
                self,
                num_tokens: int,
                max_length_class: int,
                length_auxiliary_enabled: bool,
                max_bridge_count_class: int,
                bridge_auxiliary_enabled: bool,
                boundary_auxiliary_enabled: bool,
            ) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 2)),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 2)),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 1)),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 1)),
                )
                self.rnn = nn.LSTM(
                    input_size=128 * 2,
                    hidden_size=256,
                    num_layers=2,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0.1,
                )
                self.classifier = nn.Linear(512, num_tokens)
                self.length_head = (
                    nn.Linear(512, max_length_class + 1)
                    if length_auxiliary_enabled
                    else None
                )
                self.bridge_head = (
                    nn.Linear(512, max_bridge_count_class + 1)
                    if bridge_auxiliary_enabled
                    else None
                )
                self.boundary_head = nn.Linear(512, 1) if boundary_auxiliary_enabled else None

            def forward(self, images):
                features = self.features(images)
                if features.shape[2] != 2:
                    features = features.mean(dim=2)
                else:
                    features = features.flatten(1, 2)
                sequence = features.permute(0, 2, 1)
                encoded, _ = self.rnn(sequence)
                logits = self.classifier(encoded)
                length_logits = None
                if self.length_head is not None:
                    length_logits = self.length_head(encoded.mean(dim=1))
                bridge_logits = None
                if self.bridge_head is not None:
                    bridge_logits = self.bridge_head(encoded.mean(dim=1))
                boundary_logits = None
                if self.boundary_head is not None:
                    boundary_logits = self.boundary_head(encoded).squeeze(-1)
                return {
                    "log_probs": logits.log_softmax(dim=2),
                    "length_logits": length_logits,
                    "bridge_logits": bridge_logits,
                    "boundary_logits": boundary_logits,
                }

        return Model(
            num_tokens,
            max_length_class,
            length_auxiliary_enabled,
            max_bridge_count_class,
            bridge_auxiliary_enabled,
            boundary_auxiliary_enabled,
        )


def _decode_greedy(token_sequence: list[int], token_to_char: dict[int, str]) -> str:
    """Collapse CTC repeats and blanks into text."""

    chars = []
    previous = 0
    for token in token_sequence:
        token = int(token)
        if token != 0 and token != previous:
            chars.append(token_to_char.get(token, ""))
        previous = token
    return "".join(chars)


def _decode_greedy_with_tokens(
    token_sequence: list[int],
    confidence_sequence: list[float],
    token_to_char: dict[int, str],
) -> tuple[str, list[dict]]:
    """Collapse CTC repeats/blanks while preserving emitted token evidence.

    Args:
        token_sequence: Greedy token ID at each CRNN time step.
        confidence_sequence: Probability of the greedy token at each time step.
        token_to_char: CTC token ID to Armenian glyph map.

    Returns:
        Tuple of decoded text and JSON-safe token records. Confidence is averaged
        over the contiguous timestep run that emitted the character.
    """

    text_parts = []
    records = []
    previous = 0
    run_token = None
    run_start = 0
    run_confidences: list[float] = []

    def flush_run(end_step: int) -> None:
        nonlocal previous
        if run_token is None:
            return
        token = int(run_token)
        if token != 0 and token != previous:
            char = token_to_char.get(token, "")
            if char:
                text_parts.append(char)
                records.append(
                    {
                        "index": len(records),
                        "char": char,
                        "token_id": token,
                        "confidence": float(
                            sum(run_confidences) / max(1, len(run_confidences))
                        ),
                        "time_start": int(run_start),
                        "time_end": int(end_step),
                    }
                )
        previous = token

    for step, token in enumerate(token_sequence):
        token = int(token)
        confidence = float(confidence_sequence[step])
        if run_token is None:
            run_token = token
            run_start = step
            run_confidences = [confidence]
            continue
        if token == run_token:
            run_confidences.append(confidence)
            continue
        flush_run(step - 1)
        run_token = token
        run_start = step
        run_confidences = [confidence]

    flush_run(len(token_sequence) - 1)
    return "".join(text_parts), records
