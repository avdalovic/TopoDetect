"""
SWAT Topology - Combinatorial Complex representation of SWAT system
with detailed component relationships as 1-cells
"""

import numpy as np
import toponetx as tnx
import json
import os
from src.utils.attack_utils import get_attack_indices, get_sensor_subsets, SWAT_SUB_MAP, is_actuator

class SWATComplex:
    def __init__(self, use_geco_relationships=False):
        """Initialize the SWAT combinatorial complex representation
        
        Parameters
        ----------
        use_geco_relationships : bool, default=False
            Whether to use GECO-learned relationships from SWaT.model
        """
        self.complex = tnx.CombinatorialComplex()
        self.subsystem_map = SWAT_SUB_MAP
        self.attacks, self.attack_labels = get_attack_indices("SWAT")
        self.plc_idxs, self.plc_components = get_sensor_subsets("SWAT", by_plc=True)
        self.process_idxs, _ = get_sensor_subsets("SWAT", by_plc=False)
        self.use_geco_relationships = use_geco_relationships
        
        # Store GECO relationships for later use in feature computation
        self.geco_relationships = {}
        if use_geco_relationships:
            self.geco_relationships = self._load_geco_relationships()
        
        # Build the complex
        self._build_complex()
        
    def _load_geco_relationships(self):
        """Load GECO-learned relationships from SWaT.model file"""
        print("Loading GECO relationships from SWaT.model...")
        
        # Path to the SWaT.model file
        model_path = os.path.join(os.path.dirname(__file__), "SWaT.model")
        
        if not os.path.exists(model_path):
            print(f"Warning: SWaT.model not found at {model_path}")
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
    
    def _is_sensor(self, component):
        """Check if a component is a sensor based on its naming pattern"""
        return not is_actuator("SWAT", component)
    
    def _is_actuator(self, component):
        """Check if a component is an actuator based on its naming pattern"""
        return is_actuator("SWAT", component)
    
    def _get_specific_component_relationships(self):
        """Define specific component relationships based on process knowledge"""
        relationships = [
            # Stage 1: Raw Water Stage
            ("MV101", "LIT101", "valve 101 to level sensor"),
            ("MV101", "FIT101", "valve affects flow"),
            ("FIT101", "LIT101", "flow affects level"),
            ("LIT101", "P101", "tank level influences pump"),
            ("LIT101", "P102", "tank level influences backup pump"),
            ("P101", "MV201", "pump 1 to  valve 201"),
            ("P101", "FIT201", "pump 1 to stage 2 flow meter"),
            ("P102", "FIT201", "backup pump 2 to stage 2 flow meter"),
            ("P102", "MV201", "backup pump 2 to valve 201"),
            # Stage 2: Chemical Dosing
            ("FIT201", "MV201", "flow affects valve in stage 2"),
            ("FIT201", "AIT202", "flow affects pH reading"),
            ("FIT201", "AIT201", "flow affects conductivity reading"),
            ("FIT201", "AIT203", "flow affects ORP reading"),
            ("AIT201", "P201", "conductivity controls NaCl dosing"),
            ("AIT201", "P202", "conductivity reading controls backup NaCl dosing"),
            ("AIT202", "P203", "pH reading controls HCl dosing"),
            ("AIT202", "P204", "pH reading controls backup HCl dosing"),   
            ("AIT203", "P205", "ORP reading controls NaOCl dosing"),
            ("AIT203", "P206", "ORP reading controls backup NaOCl dosing"),
            ("MV201", "LIT301", "valve affects T301 level"),
            ("MV201", "FIT301", "valve affects flow in next stage"),
            ("P201", "MV201", "NaCl flow"),
            ("P203", "MV201", "HCl flow"),
            ("P205", "MV201", "NaOCl flow"),
            ("P201", "AIT201", "NaCl dosing affects conductivity"),
            ("P202", "AIT201", "backup NaCl dosing affects conductivity"),
            ("P203", "AIT202", "HCl dosing affects pH"),
            ("P204", "AIT202", "backup HCl dosing affects pH"),
            ("P205", "AIT203", "NaOCl dosing affects ORP"),
            ("P206", "AIT203", "backup NaOCl dosing affects ORP"),

            # Stage 3: Ultrafiltration
            ("MV301", "LIT301", "valve 301 to level sensor"),
            ("MV301", "FIT301", "valve 301 to flow sensor"),
            ("LIT301", "P302", "T301 level controls pump"),
            ("LIT301", "P301", "T301 level controls  backup pump"),
            ("FIT301", "LIT301", "flow affects T301 level"),
            ("DPIT301", "MV301", "diff pressure affects UF inlet valve"),
            ("DPIT301", "MV302", "diff pressure affects outlet valve"),
            ("DPIT301", "MV303", "diff pressure triggers backwash"),
            ("DPIT301", "MV304", "pressure triggers backwash"),
            ("MV303", "FIT301", "backwash drain valve affects flow"),
            ("MV304", "FIT301", "UF drain valve affects flow"),
            ("MV302", "LIT401", "water from uf to next stage"),
            ("LIT301", "MV101", "level affects backwash flow"),
            
            # Stage 4: Dechlorination
            ("FIT401", "MV401", "flow affects valve in stage 4"),
            ("FIT401", "AIT401", "flow affects conductivity reading"),
            ("FIT401", "AIT402", "flow affects ORP reading"),
            ("FIT401", "AIT403", "flow affects pH reading"),
            ("FIT401", "AIT404", "flow affects hardness reading"),
            ("FIT401", "AIT405", "flow affects turbidity reading"),
            ("FIT401", "AIT406", "flow affects temperature reading"),
            ("LIT401", "P401", "tank level controls backup dechlorination"),
            ("LIT401", "P402", "tank level controls  dechlorination"),
            ("P402", "FIT401", "pump affects flow"),
            ("P401", "FIT401", "backup pump affects flow"),
            ("P402", "UV401", "pump to UV"),
            ("P401", "UV401", "backup pump to UV"),
            ("UV401", "AIT402", "UV affects ORP"),
            ("FIT401", "UV401", "flow affects UV operation"),
            ("AIT402", "P403", "ORP controls NaHSO3 dosing"),
            ("AIT402", "P404", "ORP controls backup NaHSO3 dosing"),
            ("AIT401", "UV401", "hardness affects UV control"),
            ("P205", "AIT402", "NaOCl affects ORP"),
            ("LIT401", "UV401", "tank level affects UV operation"),
            ("P403", "P501", "pump after UV affects RO feed pump"),
            ("P404", "P502", "pump after UV affects RO backup pump"),
            ("P403", "PIT501", "pump after UV affects pressure meter P501"),
            ("P404", "PIT501", "pump backup after UV affects pressure meter P501"),
            ("FIT401", "AIT401", "flow affects hardness reading"),
            ("UV401", "P403", "UV output triggers NaHSO3 dosing"),
            ("UV401", "P404", "UV output triggers backup NaHSO3 dosing"),
            ("AIT402", "AIT502", "ORP to ORP monitoring"),
            ("AIT401", "AIT501", "conductivity to conductivity monitoring"),

            # Stage 5: Reverse Osmosis
            ("P501", "PIT501", "pump to RO affects RO feed pressure"),
            ("P502", "PIT501", "pump backup to RO affects RO feed pressure"),
            ("P501", "AIT501", "pH monitoring"),
            ("P501", "AIT502", "ORP monitoring"),
            ("P501", "AIT503", "feed conductivity"),
            ("P501", "FIT501", "pump affects RO feed flow"),
            ("P501", "FIT502", "pump affects RO permeate flow"),
            ("P501", "FIT503", "pump affects RO reject flow"),
            ("FIT504", "P501", "recirculation flow affects pump"),
            ("AIT504", "PIT502", "permeate conductivity"),
            ("PIT503", "P602", "reject pressure affects pump"),
            ("FIT503", "PIT503", "reject flow affects pressure meter"),
            ("FIT502", "AIT504", "permeate flow affects conductivity"),
            ("FIT502", "PIT502", "permeate flow affects pressure meter"),
            ("AIT501", "P501", "pH controls RO feed pump"),
            ("AIT502", "P501", "ORP controls RO feed pump"),
            ("PIT501", "AIT501", "pressure affects pH monitoring"),
            ("PIT501", "AIT502", "pressure affects ORP monitoring"),
            ("PIT501", "AIT503", "pressure affects conductivity monitoring"),
            ("PIT502", "FIT502", "permeate pressure affects flow"),
            ("PIT503", "FIT503", "reject pressure affects flow"),
            ("FIT504", "AIT503", "recirculation affects feed conductivity"),

            # Stage 6: Backwash
            ("P602", "MV303", "pressure affects backwash pump"),
            ("P602", "MV301", "pump controls backwash flow"),
            ("FIT601", "MV301", "backwash flow affects valve control"),
            ("FIT601", "MV303", "backwash flow affects valve control"),
            ("FIT601", "MV301", "backwash flow affects valve control"),
            ("FIT601", "MV303", "backwash flow affects valve control"),
            ("P601", "FIT601", "backup pump affects backwash flow"),
            ("MV301", "MV303", "backwash valves are interconnected"),
            ("FIT601", "DPIT301", "backwash flow affects differential pressure")         
        ]
        
        return relationships
        
    def _build_complex(self):
        """Build the combinatorial complex for SWAT"""
        print("Building SWAT combinatorial complex...")
        
        # Get all unique components
        all_components = []
        for subsystem, components in self.subsystem_map.items():
            all_components.extend(components)
        
        # Remove duplicates while preserving order
        unique_components = []
        for comp in all_components:
            if comp not in unique_components:
                unique_components.append(comp)
        
        # Add components as rank 0 cells (nodes)
        print(f"Adding {len(unique_components)} components as rank 0 cells")
        for component in unique_components:
            self.complex.add_cell([component], rank=0)
        
        # Choose relationship source based on configuration
        if self.use_geco_relationships:
            print("Using GECO-learned relationships")
            relationships = self._get_geco_relationships()
        else:
            print("Using manually defined relationships")
            relationships = [(s, t, d) for s, t, d in self._get_specific_component_relationships()]
        
        print(f"Adding {len(relationships)} relationships as rank 1 cells")
        added_count = 0
        
        for relationship in relationships:
            if len(relationship) == 3:
                source, target, description = relationship
                geco_attrs = {}
            else:
                source, target, description, geco_attrs = relationship
                
            try:
                # Check if components exist
                if source in unique_components and target in unique_components:
                    # Add both components as a 1-cell (relation)
                    cell_name = f"{source}_{target}"
                    self.complex.add_cell([source, target], rank=1, name=cell_name, description=description, **geco_attrs)
                    if added_count < 10:  # Only print first 10 for brevity
                        print(f"  Added 1-cell: [{source}, {target}] ({description})")
                    elif added_count == 10:
                        print(f"  ... (suppressing further 1-cell additions)")
                    added_count += 1
                else:
                    print(f"  Warning: Could not add 1-cell [{source}, {target}] (component not found)")
            except Exception as e:
                print(f"  Error adding 1-cell [{source}, {target}]: {str(e)}")
        
        print(f"  Successfully added {added_count} component relationships")
        
        # Add PLC control relationships as rank 2 cells
        print(f"Adding PLC control groups as rank 2 cells")
        plc_added_count = 0
        
        for i, components in enumerate(self.plc_components):
            valid_components = [comp for comp in components if comp in unique_components]
            if len(valid_components) >= 2:
                cell_name = f"PLC_{i+1}"
                # Print the actual components being added
                print(f"  Attempting to add {cell_name} with components: {valid_components}")
                try:
                    self.complex.add_cell(valid_components, rank=2, name=cell_name)
                    print(f"  Added rank 2 cell: {cell_name} with {len(valid_components)} components")
                except Exception as e:
                    print(f"  Error adding {cell_name}: {str(e)}")
                plc_added_count += 1
            else:
                print(f"  Warning: Not enough valid components for PLC_{i+1}")
        
        print(f"  Successfully added {plc_added_count} PLC control groups")
        
        print(f"Complex built: {self.complex}")
    
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
    
    def get_complex(self):
        """Return the combinatorial complex"""
        return self.complex
    
    def _get_node_index_map(self):
        """Helper to create a mapping from component name to its index in the complex."""
        row_dict_01, _, _ = self.complex.incidence_matrix(0, 1, index=True)
        return {list(k)[0]: v for k, v in row_dict_01.items()}


# Example usage
def main():
    # Create the SWAT complex
    swat_complex = SWATComplex()
    
    # Print complex information
    print("\nComplex information:")
    print(swat_complex.get_complex())

if __name__ == "__main__":
    main()