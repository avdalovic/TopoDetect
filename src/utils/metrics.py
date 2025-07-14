"""
Enhanced time-aware metrics for evaluating anomaly detection performance.
Based on "Enhanced Time-series-aware Precision & Recall" by Hwang et al. (SAC '22)
and "Precision and Recall for Time Series" by Tatbul et al.
"""
import numpy as np

def calculate_standard_metrics(y_true, y_pred):
    """Calculates standard Precision, Recall, and F1-Score."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def calculate_scenario_detection(y_true, y_pred, attacks):
    """
    Calculates the fraction of attack scenarios that were correctly detected.
    An attack scenario is considered detected if at least one point within
    its time range is correctly flagged as an anomaly.
    
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
        attacks (list of tuples): List of (start_index, end_index) for each attack.

    Returns:
        float: Fraction of scenarios detected.
    """
    if not attacks:
        return 0.0
        
    detected_scenarios = 0
    total_valid_attacks = 0
    
    for start, end in attacks:
        # Check if attack indices are within bounds
        if start < len(y_true) and end < len(y_true) and start <= end:
            total_valid_attacks += 1
            # Check if there is any true positive within this attack window
            if np.any((y_pred[start:end+1] == 1) & (y_true[start:end+1] == 1)):
                detected_scenarios += 1
        else:
            print(f"Warning: Attack window [{start}, {end}] is out of bounds for data length {len(y_true)}")
            
    if total_valid_attacks == 0:
        return 0.0
        
    return detected_scenarios / total_valid_attacks

def calculate_fp_alarms(y_true, y_pred, attacks, window_seconds=60, sample_rate=1.0, original_sample_hz=1):
    """
    Counts the number of continuous false positive alarms with proper sampling rate consideration.
    An alarm is a continuous sequence of positive predictions. It is considered a
    false positive only if the entire sequence is outside of a given time
    window around any true attack.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
        attacks (list of tuples): List of (start_index, end_index) for each attack.
        window_seconds (int): The time window (in seconds) around true attacks to ignore FPs.
        sample_rate (float): The sampling rate used (e.g., 0.1 for 10% sampling).
        original_sample_hz (int): The original sampling frequency of the data before subsampling.

    Returns:
        int: The number of continuous false positive alarms.
    """
    fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
    if len(fp_indices) == 0:
        return 0

    # Calculate effective sampling frequency after subsampling
    effective_sample_hz = original_sample_hz * sample_rate
    
    # Convert time window to number of points in the sampled data
    window_points = int(window_seconds * effective_sample_hz)
    
    # Ensure minimum window size
    window_points = max(1, window_points)
    
    print(f"DEBUG FPA: window_seconds={window_seconds}, sample_rate={sample_rate}, "
          f"effective_hz={effective_sample_hz:.4f}, window_points={window_points}")
    
    # Create a "safe zone" mask around true attacks
    safe_zone = np.zeros_like(y_true, dtype=bool)
    for start, end in attacks:
        if start < len(y_true) and end < len(y_true):
            safe_start = max(0, start - window_points)
            safe_end = min(len(y_true) - 1, end + window_points)
            safe_zone[safe_start:safe_end+1] = True
    
    # Identify FPs that are outside the safe zone
    true_fp_indices = fp_indices[~safe_zone[fp_indices]]
    
    if len(true_fp_indices) == 0:
        return 0
        
    # Find contiguous blocks of these true FPs
    fp_alarms = 0
    in_alarm = False
    for i in range(len(true_fp_indices)):
        if not in_alarm:
            fp_alarms += 1
            in_alarm = True
        # Check if the next FP is not contiguous
        if i + 1 < len(true_fp_indices) and true_fp_indices[i+1] > true_fp_indices[i] + 1:
            in_alarm = False
            
    return fp_alarms

def calculate_time_aware_metrics(y_true, y_pred, attacks, theta_p=0.5, theta_r=0.1, 
                                sample_rate=1.0, original_sample_hz=1):
    """
    Calculate enhanced time-aware precision and recall (eTaP, eTaR, eTaF1).
    Based on "Enhanced Time-series-aware Precision & Recall" by Hwang et al. (SAC '22).
    
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
        attacks (list of tuples): List of (start_index, end_index) for each attack.
        theta_p (float): Threshold for precision detection score (default: 0.5).
        theta_r (float): Threshold for recall detection score (default: 0.1).
        sample_rate (float): The sampling rate used (e.g., 0.1 for 10% sampling).
        original_sample_hz (int): The original sampling frequency before subsampling.
    
    Returns:
        dict: Dictionary containing eTaP, eTaR, and eTaF1.
    """
    if not attacks:
        return {'eTaP': 0.0, 'eTaR': 0.0, 'eTaF1': 0.0}
    
    # Convert attacks to anomaly ranges for easier processing
    anomaly_ranges = []
    for start, end in attacks:
        if start < len(y_true) and end < len(y_true) and start <= end:
            anomaly_ranges.append((start, end))
    
    if not anomaly_ranges:
        return {'eTaP': 0.0, 'eTaR': 0.0, 'eTaF1': 0.0}
    
    # Find all predicted anomaly segments
    pred_segments = find_anomaly_segments(y_pred)
    
    print(f"DEBUG eTaP/eTaR: Found {len(pred_segments)} prediction segments and {len(anomaly_ranges)} attack ranges")
    print(f"DEBUG eTaP/eTaR: theta_p={theta_p}, theta_r={theta_r}, sample_rate={sample_rate}")
    
    # Calculate enhanced time-aware precision (eTaP)
    etap = calculate_enhanced_precision(pred_segments, anomaly_ranges, theta_p)
    
    # Calculate enhanced time-aware recall (eTaR)
    etar = calculate_enhanced_recall(pred_segments, anomaly_ranges, theta_r)
    
    # Calculate enhanced time-aware F1 (eTaF1)
    if etap + etar > 0:
        etaf1 = 2 * etap * etar / (etap + etar)
    else:
        etaf1 = 0.0
    
    return {'eTaP': etap, 'eTaR': etar, 'eTaF1': etaf1}

def calculate_enhanced_precision(pred_segments, anomaly_ranges, theta_p=0.5):
    """
    Calculate enhanced time-aware precision (eTaP) using the proper formula:
    eTaP = (1/|P|) * Σ_{p∈P} (s_d(p) + s_d(p) * s_p(p)) / 2
    
    Args:
        pred_segments (list): List of (start, end) tuples for predicted anomaly segments.
        anomaly_ranges (list): List of (start, end) tuples for true anomaly ranges.
        theta_p (float): Threshold for detection score.
    
    Returns:
        float: Enhanced time-aware precision.
    """
    if not pred_segments:
        return 0.0
    
    total_score = 0.0
    
    for pred_start, pred_end in pred_segments:
        pred_length = pred_end - pred_start + 1
        
        # Find the best overlapping anomaly range
        best_overlap_info = None
        max_overlap_length = 0
        
        for true_start, true_end in anomaly_ranges:
            if segments_overlap(pred_start, pred_end, true_start, true_end):
                overlap_start = max(pred_start, true_start)
                overlap_end = min(pred_end, true_end)
                overlap_length = overlap_end - overlap_start + 1
                
                if overlap_length > max_overlap_length:
                    max_overlap_length = overlap_length
                    best_overlap_info = (overlap_length, pred_length, true_start, true_end)
        
        if best_overlap_info is not None:
            overlap_length, pred_length, true_start, true_end = best_overlap_info
            
            # Calculate detection score s_d(p)
            overlap_ratio = overlap_length / pred_length
            s_d = 1.0 if overlap_ratio >= theta_p else 0.0
            
            # Calculate portion score s_p(p)
            s_p = overlap_ratio
            
            # Calculate contribution: (s_d + s_d * s_p) / 2
            contribution = (s_d + s_d * s_p) / 2.0
            total_score += contribution
        else:
            # No overlap, contribution is 0
            total_score += 0.0
    
    return total_score / len(pred_segments)

def calculate_enhanced_recall(pred_segments, anomaly_ranges, theta_r=0.1):
    """
    Calculate enhanced time-aware recall (eTaR) using the proper formula:
    eTaR = (1/|A|) * Σ_{a∈A} (s_d(a) + s_d(a) * s_p(a)) / 2
    
    Args:
        pred_segments (list): List of (start, end) tuples for predicted anomaly segments.
        anomaly_ranges (list): List of (start, end) tuples for true anomaly ranges.
        theta_r (float): Threshold for detection score.
    
    Returns:
        float: Enhanced time-aware recall.
    """
    if not anomaly_ranges:
        return 0.0
    
    total_score = 0.0
    
    for true_start, true_end in anomaly_ranges:
        true_length = true_end - true_start + 1
        
        # Find all overlapping prediction segments and calculate total coverage
        total_overlap_length = 0
        overlapping_segments = []
        
        for pred_start, pred_end in pred_segments:
            if segments_overlap(pred_start, pred_end, true_start, true_end):
                overlap_start = max(pred_start, true_start)
                overlap_end = min(pred_end, true_end)
                overlap_length = overlap_end - overlap_start + 1
                overlapping_segments.append((overlap_start, overlap_end, overlap_length))
        
        if overlapping_segments:
            # Merge overlapping segments to avoid double counting
            merged_segments = merge_overlapping_segments(overlapping_segments)
            total_overlap_length = sum(end - start + 1 for start, end, _ in merged_segments)
            
            # Calculate detection score s_d(a)
            coverage_ratio = total_overlap_length / true_length
            s_d = 1.0 if coverage_ratio >= theta_r else 0.0
            
            # Calculate portion score s_p(a)
            s_p = min(coverage_ratio, 1.0)  # Cap at 1.0
            
            # Calculate contribution: (s_d + s_d * s_p) / 2
            contribution = (s_d + s_d * s_p) / 2.0
            total_score += contribution
        else:
            # No overlap, contribution is 0
            total_score += 0.0
    
    return total_score / len(anomaly_ranges)

def merge_overlapping_segments(segments):
    """
    Merge overlapping segments to avoid double counting coverage.
    
    Args:
        segments (list): List of (start, end, length) tuples.
    
    Returns:
        list: List of merged (start, end, length) tuples.
    """
    if not segments:
        return []
    
    # Sort segments by start position
    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]
    
    for current_start, current_end, _ in sorted_segments[1:]:
        last_start, last_end, _ = merged[-1]
        
        if current_start <= last_end + 1:  # Overlapping or adjacent
            # Merge segments
            new_start = min(last_start, current_start)
            new_end = max(last_end, current_end)
            new_length = new_end - new_start + 1
            merged[-1] = (new_start, new_end, new_length)
        else:
            # Non-overlapping, add as new segment
            merged.append((current_start, current_end, current_end - current_start + 1))
    
    return merged

def find_anomaly_segments(y_pred):
    """Find continuous segments of predicted anomalies."""
    segments = []
    start = None
    
    for i, pred in enumerate(y_pred):
        if pred == 1 and start is None:
            start = i
        elif pred == 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    
    # Handle case where anomaly extends to end
    if start is not None:
        segments.append((start, len(y_pred) - 1))
    
    return segments

def segments_overlap(pred_start, pred_end, true_start, true_end):
    """Check if two segments overlap."""
    return not (pred_end < true_start or pred_start > true_end)

# Legacy functions for backward compatibility - keeping the old implementation
# but marking them as deprecated

def calculate_overlap_reward(pred_start, pred_end, true_start, true_end, alpha=0.0, cardinality='reciprocal', bias='flat'):
    """
    DEPRECATED: Legacy overlap reward calculation.
    Use calculate_enhanced_precision/recall instead.
    """
    # Find overlap region
    overlap_start = max(pred_start, true_start)
    overlap_end = min(pred_end, true_end)
    
    if overlap_start > overlap_end:
        return 0.0
    
    # Calculate basic overlap ratio
    true_length = true_end - true_start + 1
    overlap_length = overlap_end - overlap_start + 1
    overlap_ratio = overlap_length / true_length
    
    # Apply bias function if alpha > 0
    if alpha > 0:
        bias_weight = calculate_bias_weight(overlap_start, overlap_end, true_start, true_end, bias)
        return alpha * bias_weight + (1 - alpha) * overlap_ratio
    else:
        return overlap_ratio

def calculate_bias_weight(overlap_start, overlap_end, true_start, true_end, bias):
    """
    DEPRECATED: Legacy bias weight calculation.
    """
    true_length = true_end - true_start + 1
    overlap_center = (overlap_start + overlap_end) / 2
    true_center = (true_start + true_end) / 2
    
    if bias == 'flat':
        return 1.0
    elif bias == 'front':
        # Higher weight for earlier detection
        relative_pos = (overlap_center - true_start) / true_length
        return 1.0 - relative_pos
    elif bias == 'middle':
        # Higher weight for detection near center
        relative_pos = abs(overlap_center - true_center) / (true_length / 2)
        return 1.0 - min(relative_pos, 1.0)
    elif bias == 'back':
        # Higher weight for later detection
        relative_pos = (overlap_center - true_start) / true_length
        return relative_pos
    else:
        return 1.0

def calculate_existence_reward_precision(y_true, y_pred, anomaly_ranges, alpha=0.0, cardinality='reciprocal', bias='flat'):
    """
    DEPRECATED: Legacy existence reward precision calculation.
    Use calculate_enhanced_precision instead.
    """
    # Find all predicted anomaly segments
    pred_segments = find_anomaly_segments(y_pred)
    
    if not pred_segments:
        return 0.0
    
    total_reward = 0.0
    
    for pred_start, pred_end in pred_segments:
        # Check if this prediction overlaps with any true anomaly
        max_overlap_reward = 0.0
        
        for true_start, true_end in anomaly_ranges:
            if segments_overlap(pred_start, pred_end, true_start, true_end):
                # Calculate overlap reward
                overlap_reward = calculate_overlap_reward(
                    pred_start, pred_end, true_start, true_end, alpha, cardinality, bias
                )
                max_overlap_reward = max(max_overlap_reward, overlap_reward)
        
        total_reward += max_overlap_reward
    
    return total_reward / len(pred_segments)

def calculate_existence_reward_recall(y_true, y_pred, anomaly_ranges, alpha=0.0, cardinality='reciprocal', bias='flat'):
    """
    DEPRECATED: Legacy existence reward recall calculation.
    Use calculate_enhanced_recall instead.
    """
    if not anomaly_ranges:
        return 0.0
    
    total_reward = 0.0
    
    for true_start, true_end in anomaly_ranges:
        # Find all predictions that overlap with this true anomaly
        overlapping_rewards = []
        
        pred_segments = find_anomaly_segments(y_pred)
        for pred_start, pred_end in pred_segments:
            if segments_overlap(pred_start, pred_end, true_start, true_end):
                overlap_reward = calculate_overlap_reward(
                    pred_start, pred_end, true_start, true_end, alpha, cardinality, bias
                )
                overlapping_rewards.append(overlap_reward)
        
        # Aggregate rewards for this anomaly
        if overlapping_rewards:
            if cardinality == 'reciprocal':
                anomaly_reward = sum(overlapping_rewards) / len(overlapping_rewards)
            else:  # cardinality == 'one'
                anomaly_reward = max(overlapping_rewards)
        else:
            anomaly_reward = 0.0
        
        total_reward += anomaly_reward
    
    return total_reward / len(anomaly_ranges) 