"""
WADI Topology - Combinatorial Complex representation of WADI system
with detailed component relationships as 1-cells, enhanced with GECO support
"""

import numpy as np
import toponetx as tnx
from src.utils.attack_utils import get_attack_indices, get_attack_sds, WADI_SUB_MAP, is_actuator
from collections import defaultdict
from itertools import combinations
import json
import os

class WADIComplex:
    def __init__(self, component_names=None, use_geco_relationships=False):
        """Initialize the WADI combinatorial complex representation"""
        self.complex = tnx.CombinatorialComplex()
        self.subsystem_map = WADI_SUB_MAP
        self.attacks, self.attack_labels = get_attack_indices("WADI")
        self.use_geco_relationships = use_geco_relationships
        
        # Create a reverse map for component to subsystem lookup
        self.component_to_subsystem = {
            component: subsystem 
            for subsystem, components in self.subsystem_map.items() 
            for component in components
        }
        
        # Define columns to exclude from topology
        self.remove_list = [
            'Row', 'Date', 'Time', 'Attack', '2B_AIT_002_PV', '2_LS_001_AL',
            '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS',
            'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 'TOTAL_CONS_REQUIRED_FLOW'
        ]
        
        # Load GECO relationships if requested
        self.geco_relationships = {}
        if self.use_geco_relationships:
            self.geco_relationships = self._load_geco_relationships()
            print(f"Loaded {len(self.geco_relationships)} GECO relationships from WADI.model")
        
        # Build the complex
        self._build_complex(component_names)
        
    def _load_geco_relationships(self):
        """Load GECO-learned relationships from WADI.model file"""
        print("Loading GECO relationships from WADI.model...")
        
        # Path to the WADI.model file
        model_path = os.path.join(os.path.dirname(__file__), "WADI.model")
        
        if not os.path.exists(model_path):
            print(f"Warning: WADI.model not found at {model_path}")
            return {}
            
        try:
            with open(model_path, 'r') as f:
                geco_data = json.load(f)
            
            relationships = {}
            ci_data = geco_data.get('CI', {})
            
            print(f"Found {len(ci_data)} components with GECO relationships")
            
            for component, details in ci_data.items():
                if 'combination' in details and 'parameters' in details:
                    combination = details['combination']
                    parameters = details['parameters']
                    equation = details.get('equation', 'Sum')
                    error = details.get('error', 0.0)
                    threshold = details.get('threshold', 0.0)
                    
                    # Store relationship information
                    relationships[component] = {
                        'combination': combination,
                        'parameters': parameters,
                        'equation': equation,
                        'error': error,
                        'threshold': threshold,
                        'target': component  # The component being predicted
                    }
                    
                    print(f"  {component}: {equation} with {len(combination)} inputs, error={error:.6f}")
            
            return relationships
            
        except Exception as e:
            print(f"Error loading GECO relationships: {e}")
            return {}
    
    def _get_geco_relationships(self):
        """Convert GECO relationships to edge format with proper parameter handling"""
        if not self.geco_relationships:
            return []

        relationships = []
        edge_strengths = {}  # For merging duplicate edges: {edge_key: [edge_infos]}
        
        print("Processing GECO relationships with corrected parameter indexing...")
        
        for target_component, details in self.geco_relationships.items():
            combination = details['combination']
            parameters = details['parameters']
            equation = details['equation']
            error = details['error']
            threshold = details.get('threshold', 0.0)
            
            print(f"\n--- Processing {target_component} ({equation}) ---")
            print(f"Combination: {combination}")
            print(f"Parameters: {parameters}")
            print(f"Length: combination={len(combination)}, parameters={len(parameters)}")
            
            # Handle different equation types
            if equation == 'Sum':
                # Sum: b0 + Σbi·ui (self_coeff + input_coeffs + optional_bias)
                has_bias = len(parameters) > len(combination)
                param_end = len(combination)  # Don't include bias
                
                print(f"Sum equation - has_bias: {has_bias}")
                
                # Extract input coefficients (excluding self) for normalization
                input_coeffs = []
                for i, input_component in enumerate(combination):
                    if input_component != target_component and i < len(parameters):
                        coeff = parameters[i]
                        input_coeffs.append(abs(coeff))
                        print(f"  Input {input_component}: coeff={coeff:.6f}, strength={abs(coeff):.6f}")
                
                # Normalize by sum of absolute input coefficients
                total_input_strength = sum(input_coeffs) if input_coeffs else 1.0
                print(f"  Total input strength: {total_input_strength:.6f}")
                
                # Create edges for each input component
                for i, input_component in enumerate(combination):
                    if input_component != target_component and i < len(parameters):
                        raw_coeff = parameters[i]
                        raw_strength = abs(raw_coeff)
                        normalized_strength = raw_strength / total_input_strength if total_input_strength > 0 else 0.0
                        
                        self._add_edge_info(edge_strengths, input_component, target_component, 
                                          raw_coeff, raw_strength, normalized_strength, 
                                          equation, error, threshold, has_bias)
                        
            elif equation == 'Product':
                # Product: b0 + Πbi·ui 
                # Treat like Sum but with individual coefficients for each input
                print(f"Product equation - treating coefficients like Sum")
                
                # For Product equations, we have fewer parameters than combination length
                # Map available parameters to input components (excluding self)
                input_components = [comp for comp in combination if comp != target_component]
                
                if len(parameters) > 1:
                    # Available coefficients for inputs (excluding self coefficient)
                    input_coeffs = parameters[1:]  # Skip self coefficient
                    
                    print(f"  Available input coefficients: {input_coeffs}")
                    print(f"  Input components: {input_components}")
                    
                    # Extract input coefficients for normalization
                    valid_coeffs = []
                    for i, coeff in enumerate(input_coeffs):
                        if i < len(input_components):
                            valid_coeffs.append(abs(coeff))
                            print(f"    {input_components[i]}: coeff={coeff:.6f}, strength={abs(coeff):.6f}")
                    
                    # Normalize by sum of absolute coefficients (same as Sum equations)
                    total_input_strength = sum(valid_coeffs) if valid_coeffs else 1.0
                    print(f"  Total input strength: {total_input_strength:.6f}")
                    
                    # Create edges for each input component with available coefficients
                    for i, input_component in enumerate(input_components):
                        if i < len(input_coeffs):
                            raw_coeff = input_coeffs[i]
                            raw_strength = abs(raw_coeff)
                            normalized_strength = raw_strength / total_input_strength if total_input_strength > 0 else 0.0
                            
                            self._add_edge_info(edge_strengths, input_component, target_component,
                                              raw_coeff, raw_strength, normalized_strength,
                                              equation, error, threshold, False)
                            
                            print(f"    {input_component} → {target_component}: strength={raw_strength:.6f}, norm={normalized_strength:.6f}")
                        else:
                            print(f"    Warning: No coefficient available for {input_component}")
                else:
                    print(f"  Warning: Product equation has insufficient parameters")
            else:
                print(f"  Warning: Unknown equation type: {equation}")
        
        print(f"\nFound {len(edge_strengths)} unique edges before merging duplicates")
        
        # Merge duplicate edges by taking maximum strength
        merged_count = 0
        for edge_key, edge_infos in edge_strengths.items():
            if len(edge_infos) == 1:
                # Single edge
                info = edge_infos[0]
                merged_strength = info['raw_strength']
                merged_normalized = info['normalized_strength']
                direction = f"{info['source']} → {info['target']}"
                dominant_info = info
            else:
                # Multiple edges - merge by taking maximum strength
                dominant_info = max(edge_infos, key=lambda x: x['raw_strength'])
                merged_strength = max(info['raw_strength'] for info in edge_infos)
                merged_normalized = max(info['normalized_strength'] for info in edge_infos)
                
                # Create bidirectional description
                directions = [f"{info['source']} → {info['target']}" for info in edge_infos]
                direction = " & ".join(directions)
                merged_count += 1
                print(f"Merged edge {edge_key}: {len(edge_infos)} relationships, max_strength={merged_strength:.6f}")
        
            node1, node2 = edge_key
            description = f"GECO {dominant_info['equation']}: {direction} (strength={merged_strength:.4f}, error={dominant_info['error']:.6f})"
            
            relationships.append((
                node1,
                node2, 
                description,
                {
                    'geco_strength': float(merged_strength),
                    'geco_normalized_strength': float(merged_normalized),
                    'geco_equation': dominant_info['equation'],
                    'geco_error': dominant_info['error'],
                    'geco_threshold': dominant_info['threshold'],
                    'geco_direction': direction,
                    'geco_edge_count': len(edge_infos),
                    'geco_has_bias': dominant_info['has_bias']
                }
            ))
        
        print(f"Created {len(relationships)} unique edges ({merged_count} were merged from duplicates)")
        
        # Store edge lookup for efficient feature retrieval
        self._geco_edge_lookup = {}
        for node1, node2, desc, attrs in relationships:
            edge_key = tuple(sorted([node1, node2]))
            self._geco_edge_lookup[edge_key] = attrs
        
        return relationships
    
    def _add_edge_info(self, edge_strengths, input_component, target_component, 
                      raw_coeff, raw_strength, normalized_strength, equation, error, threshold, has_bias):
        """Helper method to add edge information to the edge_strengths dictionary"""
        edge_key = tuple(sorted([input_component, target_component]))
        
        edge_info = {
            'source': input_component,
            'target': target_component,
            'raw_coeff': raw_coeff,
            'raw_strength': raw_strength,
            'normalized_strength': normalized_strength,
            'equation': equation,
            'error': error,
            'threshold': threshold,
            'has_bias': has_bias
        }
        
        if edge_key not in edge_strengths:
            edge_strengths[edge_key] = []
        edge_strengths[edge_key].append(edge_info)
    
    def get_geco_edge_features(self, boundary_nodes):
        """
        Get GECO-derived features for an edge between two boundary nodes.
        
        Parameters
        ----------
        boundary_nodes : list
            List of two component names that form the edge
            
        Returns
        -------
        dict or None
            Dictionary with GECO features if relationship exists, None otherwise
        """
        if not self.use_geco_relationships or not hasattr(self, '_geco_edge_lookup'):
            return None
            
        # Create canonical edge key (sorted order)
        edge_key = tuple(sorted(boundary_nodes))
        
        # Look up in the precomputed edge lookup dictionary
        if edge_key in self._geco_edge_lookup:
            return self._geco_edge_lookup[edge_key].copy()  # Return copy to avoid modification
        
        return None
    
    def _is_sensor(self, component):
        """Check if a component is a sensor based on its naming pattern"""
        return not is_actuator("WADI", component)
    
    def _is_actuator(self, component):
        """Check if a component is an actuator based on its naming pattern"""
        return is_actuator("WADI", component)
    
    def _get_geco_component_relationships(self):
        """Get component relationships from GECO model"""
        if not self.use_geco_relationships:
            return []
        
        return self._get_geco_relationships()
    
    def _get_specific_component_relationships(self):
        """Define specific component relationships for WADI based on process knowledge,
        as described in user documentation."""
        
        # If using GECO relationships, return those instead
        if self.use_geco_relationships:
            return self._get_geco_component_relationships()
        
        # This list is based on a detailed semantic breakdown of the WADI process.
        relationships = [
            # 1. Raw_Water_Tank (Stage 1)
            # Raw water comes in through MV001, is metered by FIT001, and fills Tank1 (LT001)
            ("1_MV_001_STATUS", "1_FIT_001_PV", "inlet valve → inflow meter"),
            ("1_FIT_001_PV",   "1_LT_001_PV",  "inflow meter → tank level"),
            ("1_FIT_001_PV",   "1_AIT_001_PV",  "Inflow→conductivity"),
            ("1_LT_001_PV",    "1_AIT_001_PV",  "Level→conductivity"),
            ("1_FIT_001_PV",   "1_AIT_002_PV",  "Inflow→turbidity"),
            ("1_LT_001_PV",    "1_AIT_002_PV",  "Level→turbidity"),

            
            # When LT001 drops below low‐level, LS_001/LS_002 alarm and shut off pumps & valve
            ("1_LT_001_PV",  "1_LS_001_AL", "tank level → low level alarm 1"),
            ("1_LT_001_PV",  "1_LS_002_AL", "tank level → low level alarm 2"),
            ("1_LS_001_AL", "1_P_001_STATUS", "low level alarm → shut off pump P1"),
            ("1_LS_001_AL", "1_P_002_STATUS", "low level alarm → shut off pump P2"),
            ("1_LS_002_AL", "1_P_001_STATUS", "low level alarm 2 → shut off pump P1"),
            ("1_LS_002_AL", "1_P_002_STATUS", "low level alarm 2 → shut off pump P2"),
            ("1_LT_001_PV", "1_MV_001_STATUS", "tank level → shut off inlet valve"),
            ("1_LS_001_AL",  "1_MV_001_STATUS","alarm1 → shut off inlet valve"),
            ("1_LS_002_AL",  "1_MV_001_STATUS","alarm2 → shut off inlet valve"),


            ("1_P_005_STATUS", "2_MV_001_STATUS", "transfer pump P005 → Stage2 inlet valve"),
            ("1_P_005_STATUS", "2_MV_003_STATUS", "transfer pump P005 → Stage2 inlet valve"),
            ("1_LT_001_PV", "2_MV_003_STATUS", "tank level → Stage2 inlet valve"),
            ("1_P_005_STATUS", "2_FIT_001_PV",    "transfer pump P005 → Stage2 flow meter"),

            # Circulation Pumps (keep tank mixed)
            ("1_P_001_STATUS", "1_FIT_001_PV", "circulation pump P1 → inflow meter"),
            ("1_P_002_STATUS", "1_FIT_001_PV", "circulation pump P2 → inflow meter"),


            # 2. Elevated (Stage 2) & Cross-Subsystem links
            ("2_MV_001_STATUS", "2_FIT_001_PV", "Valve MV001 → flow sensor FIT001"),        
            ("2_MV_003_STATUS", "2_FIT_002_PV", "Valve MV003 → flow sensor FIT002"),

            # Flow → tank level  
            ("2_FIT_001_PV", "2_LT_001_PV", "Flow FIT001 → level LT001"),
            ("2_FIT_002_PV", "2_LT_002_PV", "Flow FIT002 → level LT002"),

            # Tank level → AIT
            ("2_LT_001_PV",    "2_AIT_001_PV", "Level→conductivity"),
            ("2_FIT_001_PV",   "2_AIT_001_PV", "Inflow→conductivity"), 
            ("2_LT_002_PV",    "2_AIT_002_PV", "Level→turbidity"),
            ("2_FIT_002_PV",   "2_AIT_002_PV", "Inflow→turbidity"),

            # Tank level → head pressure sensor
            ("2_LT_001_PV", "2_PIT_001_PV", "Level LT001 → head pressure PIT001"),

            # Tank levels gate the two outlet valves
            ("2_LT_001_PV", "2_MV_005_STATUS", "Level LT001 → outlet valve MV005"),
            ("2_LT_002_PV", "2_MV_006_STATUS", "Level LT002 → outlet valve MV006"),

            #2. Booster (Stage 2)

            # Outlet valves → booster pumps to Consumers
            ("2_MV_005_STATUS", "2_P_003_STATUS", "Valve MV005 → pump P003"),
            ("2_MV_006_STATUS", "2_P_004_STATUS", "Valve MV006 → pump P004"),


            # Discharge pressure → PIC003 input
            ("2_PIT_002_PV", "2_PIC_003_PV", "Discharge pressure → PIC003 input"),
            # Suction pressure → PIC003 input
            ("2_PIT_003_PV", "2_PIC_003_PV", "Suction pressure → PIC003 input"),
            ("2_PIC_003_PV", "2_PIC_003_CO", "PIC003 process value → control output"),
            ("2_PIC_003_CO", "2_P_003_SPEED", "Controller output sets pump speed"),
            ("2_PIC_003_CO", "2_P_004_SPEED", "Controller output sets pump speed"),
            ("2_PIC_003_CO", "2_MCV_007_CO", "Controller output drives booster valve"),
            ("2_P_003_SPEED", "2_DPIT_001_PV", "Pump speed affects differential pressure"),
            ("2_P_004_SPEED", "2_DPIT_001_PV", "Pump speed affects differential pressure"),
            ("2_P_003_STATUS", "2_DPIT_001_PV", "Pump status affects differential pressure"),
            ("2_P_004_STATUS", "2_DPIT_001_PV", "Pump status affects differential pressure"),
            ("2_P_003_SPEED", " 2_PIT_002_PV", "Pump speed affects discharge pressure"),
            ("2_P_004_SPEED", " 2_PIT_002_PV", "Pump speed affects discharge pressure"),

            ("2_PIC_003_SP",  "2_PIC_003_PV",  "Setpoint→process value"),
            ("2_PIC_003_SP",  "2_PIC_003_CO",  "Setpoint→control output"),


            


            # 3. Return (Stage 3) & Cross-Subsystem links
            # Return inflow affects return tank level
            ("3_FIT_001_PV", "3_LT_001_PV", "Return inflow affects return tank level"),
            # Return tank level and alarms
            ("3_LT_001_PV", "3_LS_001_AL", "Return tank level triggers low-level alarm"),
            # Return tank level controls return pumps
            ("3_LT_001_PV", "3_P_001_STATUS", "Return level controls return pump P1"),
            ("3_LT_001_PV", "3_P_002_STATUS", "Return level controls return pump P2"),
            ("3_LT_001_PV", "3_P_003_STATUS", "Return level controls return pump P3"),
            ("3_LT_001_PV", "3_P_004_STATUS", "Return level controls return pump P4"),
            # Return pump isolation valves
            ("3_P_001_STATUS", "3_FIT_001_PV","Pump P001 → return flow meter FIT001"),
            ("3_P_002_STATUS", "3_FIT_001_PV", "Pump P002 → return flow meter FIT001"),

            ("3_P_003_STATUS", "3_MV_002_STATUS", "Pump P003 → backwash valve MV002"),
            ("3_P_004_STATUS", "3_MV_002_STATUS", "Pump P004 → backwash valve MV002"),
            ("3_MV_002_STATUS", "1_FIT_001_PV", "Backwash valve MV002 → Stage1 inflow meter FIT001"),
            ("3_P_003_STATUS", "1_FIT_001_PV", "Pump P003 → Stage1 inflow meter FIT001"),
            ("3_P_004_STATUS", "1_FIT_001_PV", "Pump P004 → Stage1 inflow meter FIT001")
        ]

        # ─── Stage 1: Raw‐Water Tank → Quality Sensors ───
        # For each 1_AIT_00x, connect both the inflow meter and the tank‐level sensor:

        for ait in ["001","002","003","004","005"]:
            relationships.extend([
                # Inflow → quality
                (f"1_FIT_001_PV", f"1_AIT_{ait}_PV",
                 f"Inflow FIT001 → quality sensor AIT{ait}"),
                # Level  → quality
                (f"1_LT_001_PV",  f"1_AIT_{ait}_PV",
                 f"Tank level LT001 → quality sensor AIT{ait}")
            ])


        # ─── Stage 2: Elevated Reservoir → Quality Sensors ───
        # For each 2A_AIT_00x, connect the two inflow FITs and the two level LTs:

        for ait in ["001","002","003","004"]:
            relationships.extend([
                # FIT001 → quality
                (f"2_FIT_001_PV", f"2A_AIT_{ait}_PV",
                 f"Inflow FIT001 → quality sensor 2A_AIT{ait}"),
                # FIT002 → quality
                (f"2_FIT_002_PV", f"2A_AIT_{ait}_PV",
                 f"Inflow FIT002 → quality sensor 2A_AIT{ait}"),
                # LT001  → quality
                (f"2_LT_001_PV",  f"2A_AIT_{ait}_PV",
                 f"Level LT001 → quality sensor 2A_AIT{ait}"),
                # LT002  → quality
                (f"2_LT_002_PV",  f"2A_AIT_{ait}_PV",
                 f"Level LT002 → quality sensor 2A_AIT{ait}")
            ])

        
        # ─── Consumer Lines (Branches 101…601) ───

        consumer_ids = ["101","201","301","401","501","601"]
        for cid in consumer_ids:
            # Flow controller loop
            relationships.append(
                (f"2_FIC_{cid}_PV", f"2_FIC_{cid}_CO",
                 f"Consumer {cid}: flow PV → controller CO"))
            relationships.append(
                (f"2_FIC_{cid}_CO", f"2_MCV_{cid}_CO",
                 f"Consumer {cid}: controller CO → modulating valve MCV{cid}"))
            relationships.append(
                (f"2_MCV_{cid}_CO", f"2_FIC_{cid}_PV",
                 f"Consumer {cid}: valve MCV{cid} → flow PV"))
            
            # Main elevated‐tank feed valve to this branch
            relationships.append(
                (f"2_MV_{cid}_STATUS", f"2_FIC_{cid}_PV",
                 f"Elevated‐tank valve MV{cid} → branch flow PV"))
            
            # Level‐switch interlocks on isolation valve (MV)
            relationships.append(
                (f"2_LS_{cid}_AH", f"2_MV_{cid}_STATUS",
                 f"Consumer {cid}: high level trips MV{cid}"))
            relationships.append(
                (f"2_LS_{cid}_AL", f"2_MV_{cid}_STATUS",
                 f"Consumer {cid}: low level trips MV{cid}"))
            
            # Level‐switch to drain solenoid (SV)
            relationships.append(
                (f"2_LS_{cid}_AH", f"2_SV_{cid}_STATUS",
                 f"Consumer {cid}: high level opens drain SV{cid}"))
            relationships.append(
                (f"2_LS_{cid}_AL", f"2_SV_{cid}_STATUS",
                 f"Consumer {cid}: low level may trigger SV{cid}"))
            
            # Drain solenoid → return‐line flow meter in Stage 3
            relationships.append(
                (f"2_SV_{cid}_STATUS", "3_FIT_001_PV",
                 f"SV{cid} → return‐line flow FIT001"))
            
            # Consumer flow totalizer (FQ)
            relationships.append(
                (f"2_FIC_{cid}_PV", f"2_FQ_{cid}_PV",
                 f"Consumer {cid}: flow PV → totalizer FQ{cid}"))
            relationships.append(
                (f"2_FQ_{cid}_PV", f"2_FIC_{cid}_PV",
                 f"Consumer {cid}: totalizer FQ{cid} → flow PV"))

        return relationships
        
    def get_filtered_components(self, component_names=None):
        """
        Get the list of components after filtering out problematic features.
        
        Parameters
        ----------
        component_names : list, optional
            List of component names to filter. If None, uses all components from subsystem map.
            
        Returns
        -------
        list
            Filtered list of component names
        """
        if component_names is None:
            # Generate from subsystem map
            all_components = []
            for components in self.subsystem_map.values():
                all_components.extend(components)
            component_names = all_components
        
        # Filter components and remove duplicates, preserving order
        filtered_components = []
        for comp in component_names:
            if comp not in self.remove_list and comp not in filtered_components:
                filtered_components.append(comp)
        
        print(f"Filtered {len(component_names)} components to {len(filtered_components)} (removed {len(component_names) - len(filtered_components)})")
        return filtered_components

    def _build_complex(self, component_names=None):
        """Build the combinatorial complex for WADI"""
        print("Building WADI combinatorial complex...", flush=True)
        
        if self.use_geco_relationships:
            print("Using GECO relationships from WADI.model", flush=True)
        else:
            print("Using manually defined relationships", flush=True)
        
        # Filter components using remove_list
        unique_components = self.get_filtered_components(component_names)
        
        if component_names is None:
            print("No component list provided, building from full topology.", flush=True)
        else:
            print(f"Building complex with a provided list of {len(component_names)} components.", flush=True)
        
        print(f"After filtering: using {len(unique_components)} components", flush=True)
        
        # 1. Add components as rank 0 cells (nodes)
        print(f"Adding {len(unique_components)} components as rank 0 cells (nodes)...", flush=True)
        for component in unique_components:
            self.complex.add_cell([component], rank=0)
            
        # 2. Add specific component relationships as rank 1 cells (edges)
        relationships = self._get_specific_component_relationships()
        print(f"Attempting to add {len(relationships)} component relationships as rank 1 cells (edges)...", flush=True)
        added_count = 0
        
        for relationship in relationships:
            if len(relationship) == 3:
                source, target, description = relationship
                geco_attrs = {}
            else:
                source, target, description, geco_attrs = relationship
                
            try:
                # Check if components exist and are not in remove_list
                if (source in unique_components and target in unique_components and 
                    source not in self.remove_list and target not in self.remove_list):
                    # Add both components as a 1-cell (relation)
                    cell_name = f"{source}_{target}"
                    self.complex.add_cell([source, target], rank=1, name=cell_name, description=description, **geco_attrs)
                    if added_count < 10:  # Only print first 10 for brevity
                        print(f"  Added 1-cell: [{source}, {target}] ({description})")
                    elif added_count == 10:
                        print(f"  ... (suppressing further 1-cell additions)")
                    added_count += 1
                else:
                    if source in self.remove_list or target in self.remove_list:
                        # Silently skip relationships involving removed components
                        continue
                    else:
                        print(f"  Warning: Could not add 1-cell [{source}, {target}] (component not found)")
            except Exception as e:
                print(f"  Error adding 1-cell [{source}, {target}]: {str(e)}")
        
        print(f"Successfully added {added_count} unique component relationships.", flush=True)
            
        # 3. Add subsystems as rank 2 cells (faces)
        print(f"Adding {len(self.subsystem_map)} subsystems as rank 2 cells (faces)...", flush=True)
        subsystem_added_count = 0
        for subsystem, components in self.subsystem_map.items():
            # Filter the components for this subsystem to only those that exist in the complex and are not removed
            valid_components = [comp for comp in components if comp in unique_components and comp not in self.remove_list]
            if len(valid_components) >= 2:
                self.complex.add_cell(valid_components, rank=2, name=subsystem)
                subsystem_added_count += 1
                print(f"  Added subsystem '{subsystem}' with {len(valid_components)} components")
            else:
                print(f"  Warning: Not enough valid components for subsystem '{subsystem}' to form a 2-cell ({len(valid_components)} components).", flush=True)
        print(f"Successfully added {subsystem_added_count} subsystems.", flush=True)
            
        print(f"Complex built: {self.complex}", flush=True)
    
    def get_complex(self):
        """Return the combinatorial complex"""
        return self.complex



# Example usage
def main():
    # Create the WADI complex
    wadi_complex = WADIComplex()
    
    # Print complex information
    print("\nComplex information:", flush=True)
    print(f"Complex: {wadi_complex.get_complex()}", flush=True)

if __name__ == "__main__":
    main()