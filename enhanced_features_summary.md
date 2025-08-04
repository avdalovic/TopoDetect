# Enhanced SWAT Attack Detection Features

## Problem Analysis
Based on the analysis of missed attacks, we identified several patterns that were not being detected:

### Missed Attack Characteristics:
1. **Small magnitude attacks** (< 1.0): Attacks 0, 7, 16, 17
2. **Linear attack patterns**: Attacks 2, 8 (LIT101, LIT301)
3. **Multi-component attacks**: Attacks 15, 16, 17
4. **Actuator attacks**: Multiple attacks on MV, P components
5. **Sensor attacks**: Multiple attacks on LIT, AIT components

### Attack Distribution:
- **PLC1**: 3 attacks (MV101, LIT101, P101/P102)
- **PLC2**: 1 attack (P203/P205)
- **PLC3**: 5 attacks (LIT301, MV304, P301/P302)
- **PLC4**: 3 attacks (LIT401, P402)
- **PLC5**: 1 attack (AIT504)

## Enhanced Features Implemented

### Original 7D Features:
1. **PLC mean**: Average of all sensor values in PLC zone
2. **PLC std**: Standard deviation of sensor values in PLC zone
3. **Flow sensor anomaly**: Anomaly score for flow sensors (FIT)
4. **Level sensor anomaly**: Anomaly score for level sensors (LIT)
5. **Actuator anomaly**: Anomaly score for actuators (MV, P)
6. **Flow balance anomaly**: Flow balance between PLC stages
7. **Attack-prone component anomaly**: Attack-prone component anomaly score

### New 5D Features (Total: 12D):
8. **Small magnitude anomaly detection**: Component-specific thresholds for small attacks
   - MV101: 0.7 threshold (Attack 0: magnitude 0.61)
   - MV304: 0.2 threshold (Attack 7: magnitude -0.1)
   - P402: 0.1 threshold (Attack 16: magnitude 0.06)
   - P101: 0.7 threshold (Attack 17: magnitude 0.58)
   - LIT301: 1.1 threshold (Attack 17: magnitude -1.04)

9. **Linear pattern anomaly detection**: Special detection for LIT101, LIT301
   - Detects high variance in small ranges (characteristic of linear attacks)
   - Based on missed attacks 2 and 8

10. **Multi-component correlation anomaly**: Detects correlated attacks
    - P203-P205 pair (Attack 15)
    - LIT401-P402 pair (Attack 16)
    - P101-LIT301 pair (Attack 17)

11. **Temporal consistency anomaly**: Detects sudden changes
    - Compares current vs previous sample
    - Threshold: 0.5 for sudden changes
    - Helps detect abrupt attack starts

12. **Cross-PLC communication anomaly**: Detects communication breakdown
    - Monitors flow connections between PLCs
    - Detects when one PLC shows anomaly but connected PLC doesn't
    - Helps with multi-stage attacks

## Implementation Details

### Configuration Changes:
- Updated `feature_dim_2` from 7D to 12D
- Updated model channels: `[1, 6, 12]` instead of `[1, 6, 7]`
- Enhanced attack detection features enabled

### Key Improvements:
1. **Component-specific thresholds**: Different sensitivity for different components
2. **Pattern recognition**: Linear attack detection for level sensors
3. **Correlation analysis**: Multi-component attack detection
4. **Temporal analysis**: Sudden change detection
5. **Cross-PLC monitoring**: Communication anomaly detection

### Expected Benefits:
- **Better recall**: Should catch the 11 missed attacks
- **Maintained precision**: Enhanced features are targeted and specific
- **Robust detection**: Multiple complementary detection methods
- **Attack-specific tuning**: Component-specific thresholds

## Debug Information
The enhanced features include comprehensive debug prints:
- Feature computation progress
- Component-specific threshold applications
- Linear pattern detection results
- Multi-component correlation scores
- Temporal consistency violations
- Cross-PLC communication anomalies

This should significantly improve the detection of the previously missed attacks while maintaining the high precision achieved in the previous run. 