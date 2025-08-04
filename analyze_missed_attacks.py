#!/usr/bin/env python3
"""
Analyze missed SWAT attacks to understand detection failures.
"""

import numpy as np
from src.utils.attack_utils import get_attack_indices, get_attack_sds

def analyze_missed_attacks():
    """Analyze the characteristics of missed attacks."""
    
    # Get all attack information
    attacks, true_labels = get_attack_indices("SWAT")
    attack_sds = get_attack_sds("SWAT")
    
    # Missed attacks from user's list
    missed_attack_indices = [0, 2, 4, 7, 8, 11, 15, 16, 17, 19, 21]  # Removed 8.5 as it's not a valid index
    
    print("=== MISSED ATTACKS ANALYSIS ===")
    print()
    
    # Analyze each missed attack
    for attack_idx in missed_attack_indices:
        if attack_idx >= len(attacks):
            continue
            
        attack_range = attacks[attack_idx]
        attack_label = true_labels[attack_idx]
        
        # Find corresponding attack details
        attack_details = []
        for sd in attack_sds:
            if sd[0] == attack_idx:
                attack_details.append(sd)
        
        print(f"Attack {attack_idx} ({attack_idx+1} in Doc):")
        print(f"  Range: {attack_range[0]}-{attack_range[-1]} ({len(attack_range)} points)")
        print(f"  Target components: {attack_label}")
        print(f"  Attack details:")
        
        for detail in attack_details:
            attack_id, component, attack_type, scope, magnitude = detail
            print(f"    - {component}: {attack_type} attack, {scope} scope, magnitude {magnitude}")
        
        # Analyze attack characteristics
        print(f"  Characteristics:")
        
        # Check if it's a multi-component attack
        if len(attack_label) > 1:
            print(f"    - Multi-component attack affecting {len(attack_label)} components")
        
        # Check attack type patterns
        attack_types = [detail[2] for detail in attack_details]
        if 'line' in attack_types:
            print(f"    - Linear attack pattern (harder to detect)")
        
        # Check magnitude patterns
        magnitudes = [detail[4] for detail in attack_details]
        small_magnitudes = [m for m in magnitudes if abs(m) < 1.0]
        if small_magnitudes:
            print(f"    - Small magnitude attacks ({small_magnitudes}) - may be below threshold")
        
        # Check component types
        component_types = []
        for component in attack_label:
            if 'IT' in component:
                component_types.append('sensor')
            else:
                component_types.append('actuator')
        
        print(f"    - Component types: {list(set(component_types))}")
        
        print()
    
    # Summary analysis
    print("=== SUMMARY ANALYSIS ===")
    
    # Count attack types
    all_missed_details = []
    for attack_idx in missed_attack_indices:
        for sd in attack_sds:
            if sd[0] == attack_idx:
                all_missed_details.append(sd)
    
    attack_types = [detail[2] for detail in all_missed_details]
    attack_scopes = [detail[3] for detail in all_missed_details]
    magnitudes = [detail[4] for detail in all_missed_details]
    components = [detail[1] for detail in all_missed_details]
    
    print(f"Total missed attacks: {len(missed_attack_indices)}")
    print(f"Attack types: {dict(zip(*np.unique(attack_types, return_counts=True)))}")
    print(f"Attack scopes: {dict(zip(*np.unique(attack_scopes, return_counts=True)))}")
    print(f"Component types affected:")
    
    component_type_counts = {}
    for component in components:
        if 'IT' in component:
            comp_type = 'sensor'
        else:
            comp_type = 'actuator'
        component_type_counts[comp_type] = component_type_counts.get(comp_type, 0) + 1
    
    print(f"  {component_type_counts}")
    
    # Magnitude analysis
    small_mag = len([m for m in magnitudes if abs(m) < 1.0])
    medium_mag = len([m for m in magnitudes if 1.0 <= abs(m) < 10.0])
    large_mag = len([m for m in magnitudes if abs(m) >= 10.0])
    
    print(f"Magnitude distribution:")
    print(f"  Small (<1.0): {small_mag}")
    print(f"  Medium (1.0-10.0): {medium_mag}")
    print(f"  Large (>=10.0): {large_mag}")
    
    # PLC zone analysis
    plc_zones = {
        'PLC1': ['MV101', 'LIT101', 'FIT101', 'P101', 'P102'],
        'PLC2': ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206'],
        'PLC3': ['FIT301', 'LIT301', 'DPIT301', 'P301', 'P302', 'MV301', 'MV302', 'MV303', 'MV304'],
        'PLC4': ['UV401', 'P401', 'P402', 'P403', 'P404', 'AIT401', 'AIT402', 'FIT401', 'LIT401'],
        'PLC5': ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503']
    }
    
    plc_attack_counts = {plc: 0 for plc in plc_zones.keys()}
    for component in components:
        for plc, plc_comps in plc_zones.items():
            if component in plc_comps:
                plc_attack_counts[plc] += 1
                break
    
    print(f"PLC zone distribution:")
    for plc, count in plc_attack_counts.items():
        if count > 0:
            print(f"  {plc}: {count} attacks")

if __name__ == "__main__":
    analyze_missed_attacks() 