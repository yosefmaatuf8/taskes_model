from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
from scipy.optimize import linear_sum_assignment
import numpy as np
from divide_sentences import generate_rttm_from_file, load_model, load_dotenv
import os

import io

def load_rttm(rttm_source):
    """Load RTTM file content into an Annotation object."""
    annotation = Annotation()
    speakers = set()

    # Check if input is a file path or raw RTTM string
    if isinstance(rttm_source, str) and "\n" in rttm_source:
        file = io.StringIO(rttm_source)  # Treat string as a file
    else:
        file = open(rttm_source, "r")  # Open as a file

    with file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
                speakers_labels = speaker.split(".")
                for skp in speakers_labels:
                    annotation[Segment(start, start + duration)] = skp
                    if skp != "unknow":
                        speakers.add(skp)
    return annotation, list(speakers)

def evaluate_der(model_rttm, truth_rttm):
    """Compute Diarization Error Rate (DER) with penalties for unnecessary speakers, unknown segments, and missing truth speakers, with specific handling for multiple model speakers matching one truth speaker."""
    reference, truth_speakers = load_rttm(truth_rttm)
    hypothesis, model_speakers = load_rttm(model_rttm)

    cost_matrix = np.zeros((len(truth_speakers), len(model_speakers)))

    for i, true_spk in enumerate(truth_speakers):
        for j, model_spk in enumerate(model_speakers):
            cost_matrix[i, j] = sum(
                1 for seg, _, spk in hypothesis.itertracks(yield_label=True)
                if spk == model_spk and true_spk in reference.get_labels(seg)
            )

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapping = {model_speakers[j]: truth_speakers[i] for i, j in zip(row_ind, col_ind)}

    mapped_hypothesis = Annotation()
    unnecessary_speakers = set()  # Track extra speakers
    unknown_segments = 0  # Track segments that don't align
    missing_truth_speakers = set(truth_speakers)  # Track missing truth speakers
    truth_speaker_mapped = set()  # Track truth speakers that have been mapped to model speakers

    for seg, _, spk in hypothesis.itertracks(yield_label=True):
        mapped_spk = mapping.get(spk, None)

        if mapped_spk:
            mapped_hypothesis[seg] = mapped_spk
            missing_truth_speakers.discard(mapped_spk)  # Remove mapped truth speakers from the set of missing speakers
            truth_speaker_mapped.add(mapped_spk)  # Mark this truth speaker as mapped
        else:
            unnecessary_speakers.add(spk)  # Speaker not mapped
            unknown_segments += 1  # Segment does not match reference

    # Calculate DER
    metric = DiarizationErrorRate()
    der_score = metric(reference, mapped_hypothesis)

    # Penalties
    penalty_unnecessary_speakers = len(unnecessary_speakers) * 0.0005  # 5% penalty per extra speaker
    penalty_unknown_segments = unknown_segments * 0.0001  # 1% penalty per unmatched segment
    
    # Apply penalty for missing truth speakers only once per truth speaker
    missing_truth_penalty_count = len(missing_truth_speakers)
    penalty_missing_truth_speakers = missing_truth_penalty_count * 0.005  # 50% penalty per missing truth speaker

    total_der = der_score + penalty_unnecessary_speakers + penalty_unknown_segments + penalty_missing_truth_speakers

    return total_der, mapping, len(unnecessary_speakers), unknown_segments, missing_truth_penalty_count

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
inference = load_model(api_key)

# Run evaluation for multiple thresholds
thresholds_closest = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
thresholds_similarity = [0.1 ,0.2, 0.3, 0.4, 0.5, 0.6]
audio_file_path = "../meeting/audio1725163871"
truth_rttm = "../meeting/audio1725163871_truth.rttm"

results = {}

for threshold_closest in thresholds_closest:
    for threshold_similarity in thresholds_similarity:
        print(f"Running model with threshold {threshold_closest}/{threshold_similarity}...")

        # Get RTTM content as a string
    # Ensure model_rttm_content is a string
        model_rttm_content = f"{audio_file_path}_{threshold_closest}_{threshold_similarity}.rttm"
        if not os.path.exists(model_rttm_content):
            generate_rttm_from_file(audio_file_path, inference, threshold_closest, threshold_similarity)
        # Evaluate DER using RTTM string (no need for file path)
        der, mapping, extra_speakers, unknown_segs, missing_truth_speakers = evaluate_der(model_rttm_content, truth_rttm)
        results[f"{threshold_closest}/{threshold_similarity}"] = der

        print(f"Threshold {threshold_closest}/{threshold_similarity} → DER: {der:.2%}, "
            f"Mapping: {mapping}, Extra Speakers: {extra_speakers}, Unknown Segments: {unknown_segs}, "
            f"Missing Truth Speakers: {missing_truth_speakers}")
# Find best threshold
best_threshold = min(results, key=results.get)
print(f"\nBest threshold: {best_threshold} with DER {results[best_threshold]:.2%}")

