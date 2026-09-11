import argparse
import re


def average_or_na(values, transform=lambda x: x):
    if not values:
        return "N/A"
    return transform(float(sum(values)) / len(values))


def _extract_with_patterns(line, patterns, transform=float):
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return transform(match.group(1))
    return None


def extract_peak_gpu_memory_gb(line):
    value = _extract_with_patterns(
        line,
        [r"^Peak GPUMemory \(GB\): ([\d.]+)"],
        float,
    )
    if value is not None:
        return value

    legacy_bytes = _extract_with_patterns(
        line,
        [r"^after generation memory: (\d+)"],
        int,
    )
    if legacy_bytes is not None:
        return float(legacy_bytes) / (1024 * 1024 * 1024)
    return None


def extract_total_time_ms(line):
    return _extract_with_patterns(
        line,
        [
            r"^TotalTime \(ms\): ([\d.]+)",
            r"^Generation latency time is: ([\d.]+)",
        ],
        float,
    )


def extract_prefill_time_ms(line):
    return _extract_with_patterns(
        line,
        [
            r"^PrefillTime \(ms\): ([\d.]+)",
            r"^Generation prefill time is: ([\d.]+)",
        ],
        float,
    )


def extract_vision_encoder_time_ms(line):
    return _extract_with_patterns(
        line,
        [r"^VisionEncoderTime \(ms\): ([\d.]+)"],
        float,
    )


def extract_selector_score_select_time_ms(line):
    return _extract_with_patterns(
        line,
        [r"^SelectorScoreSelectTime \(ms\): ([\d.]+)"],
        float,
    )


def extract_llm_prefill_time_ms(line):
    return _extract_with_patterns(
        line,
        [r"^LLMPrefillTime \(ms\): ([\d.]+)"],
        float,
    )


def extract_kv_cache_mb(line):
    return _extract_with_patterns(
        line,
        [r"^KVCache \(MB\): ([\d.]+)"],
        float,
    )


def extract_retained_visual_tokens(line):
    return _extract_with_patterns(
        line,
        [
            r"^Retained Visual Tokens: ([\d.]+)",
            r"^Input visual token number is: ([\d.]+)",
        ],
        float,
    )


def process_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    peak_gpu_memory_gb_array = []
    total_time_ms_array = []
    prefill_time_ms_array = []
    vision_encoder_time_ms_array = []
    selector_score_select_time_ms_array = []
    llm_prefill_time_ms_array = []
    kv_cache_mb_array = []
    retained_visual_tokens_array = []

    for line in lines:
        peak_gpu_memory_gb = extract_peak_gpu_memory_gb(line)
        if peak_gpu_memory_gb is not None:
            peak_gpu_memory_gb_array.append(peak_gpu_memory_gb)

        total_time_ms = extract_total_time_ms(line)
        if total_time_ms is not None:
            total_time_ms_array.append(total_time_ms)

        prefill_time_ms = extract_prefill_time_ms(line)
        if prefill_time_ms is not None:
            prefill_time_ms_array.append(prefill_time_ms)

        vision_encoder_time_ms = extract_vision_encoder_time_ms(line)
        if vision_encoder_time_ms is not None:
            vision_encoder_time_ms_array.append(vision_encoder_time_ms)

        selector_score_select_time_ms = extract_selector_score_select_time_ms(line)
        if selector_score_select_time_ms is not None:
            selector_score_select_time_ms_array.append(selector_score_select_time_ms)

        llm_prefill_time_ms = extract_llm_prefill_time_ms(line)
        if llm_prefill_time_ms is not None:
            llm_prefill_time_ms_array.append(llm_prefill_time_ms)

        kv_cache_mb = extract_kv_cache_mb(line)
        if kv_cache_mb is not None:
            kv_cache_mb_array.append(kv_cache_mb)

        retained_visual_tokens = extract_retained_visual_tokens(line)
        if retained_visual_tokens is not None:
            retained_visual_tokens_array.append(retained_visual_tokens)

    return (
        prefill_time_ms_array,
        total_time_ms_array,
        peak_gpu_memory_gb_array,
        vision_encoder_time_ms_array,
        selector_score_select_time_ms_array,
        llm_prefill_time_ms_array,
        kv_cache_mb_array,
        retained_visual_tokens_array,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input path arg")
    parser.add_argument("--path", type=str, help="path to the log", default="./log_eval.log")
    args = parser.parse_args()

    (
        prefill_array,
        total_time_array,
        peak_gpu_memory_array,
        vision_encoder_array,
        selector_score_select_array,
        llm_prefill_array,
        kv_cache_array,
        retained_visual_tokens_array,
    ) = process_file(args.path)
    print(f"Average PrefillTime (ms): {average_or_na(prefill_array)}")
    print(f"Average TotalTime (ms): {average_or_na(total_time_array)}")
    print(f"Average Peak GPUMemory (GB): {average_or_na(peak_gpu_memory_array)}")
    print(f"Average VisionEncoderTime (ms): {average_or_na(vision_encoder_array)}")
    print(f"Average SelectorScoreSelectTime (ms): {average_or_na(selector_score_select_array)}")
    print(f"Average LLMPrefillTime (ms): {average_or_na(llm_prefill_array)}")
    print(f"Average KVCache (MB): {average_or_na(kv_cache_array)}")
    print(f"Average Retained Visual Tokens: {average_or_na(retained_visual_tokens_array)}")
