"""
TEP Topology - Combinatorial Complex representation of Tennessee Eastman Process (TEP)
with detailed component relationships as 1-cells based on process knowledge
"""

import numpy as np
import toponetx as tnx
from src.utils.attack_utils import get_attack_indices, get_sensor_subsets, TEP_SUB_MAP, TEP_COLUMN_NAMES, is_actuator

class TEPComplex:
    def __init__(self):
        """Initialize the TEP combinatorial complex representation"""
        self.complex = tnx.CombinatorialComplex()
        self.subsystem_map = TEP_SUB_MAP
        self.column_names = TEP_COLUMN_NAMES
        self.attacks, self.attack_labels = get_attack_indices("TEP")
        self.plc_idxs, self.plc_components = get_sensor_subsets("TEP", by_plc=True)
        
        # Build the complex
        self._build_complex()
        
    def _is_sensor(self, component):
        """Check if a component is a sensor based on its naming pattern"""
        return not is_actuator("TEP", component)
    
    def _is_actuator(self, component):
        """Check if a component is an actuator based on its naming pattern"""
        return is_actuator("TEP", component)
    
    def _get_specific_component_relationships(self):
        """
        Define specific component relationships based on Tennessee Eastman Process knowledge.
        
        The TEP is a chemical process with:
        - 5 feed streams (A, D, E, A&C combined, Recycle)
        - Reactor with temperature/pressure/level control
        - Product separator with temperature/pressure/level control  
        - Stripper with level/pressure/temperature control
        - Compressor for recycle stream
        - Various composition analyzers
        - 12 manipulated variables (actuators)
        """
        relationships = [
            # === FEED SYSTEM ===
            # Feed flows affect reactor feed rate
            ("A Feed", "Reactor Feed Rate", "A feed contributes to reactor feed"),
            ("D Feed", "Reactor Feed Rate", "D feed contributes to reactor feed"),
            ("E Feed", "Reactor Feed Rate", "E feed contributes to reactor feed"),
            ("A and C Feed", "Reactor Feed Rate", "A&C feed contributes to reactor feed"),
            ("Recycle Flow", "Reactor Feed Rate", "Recycle contributes to reactor feed"),
            
            # Feed control loops - actuators control feed flows
            ("A Feed (MV)", "A Feed", "A feed valve controls A feed flow"),
            ("D Feed (MV)", "D Feed", "D feed valve controls D feed flow"),
            ("E Feed (MV)", "E Feed", "E feed valve controls E feed flow"),
            ("A and C Feed (MV)", "A and C Feed", "A&C feed valve controls A&C feed flow"),
            ("Recycle (MV)", "Recycle Flow", "Recycle valve controls recycle flow"),
            
            # === REACTOR SYSTEM ===
            # Reactor feed rate affects reactor conditions
            ("Reactor Feed Rate", "Reactor Pressure", "Feed rate affects reactor pressure"),
            ("Reactor Feed Rate", "Reactor Level", "Feed rate affects reactor level"),
            ("Reactor Feed Rate", "Reactor Temperature", "Feed rate affects reactor temperature"),
            
            # Reactor level control
            ("Reactor Level", "Product Sep Level", "Reactor level affects separator feed"),
            ("Reactor Level", "Product Sep Pressure", "Reactor level affects separator pressure"),
            
            # Reactor temperature control loop
            ("Reactor Temperature", "Reactor Coolant Temp", "Reactor temp affects coolant temp"),
            ("Reactor Coolant (MV)", "Reactor Coolant Temp", "Coolant valve controls coolant temp"),
            ("Reactor Coolant Temp", "Reactor Temperature", "Coolant temp controls reactor temp"),
            
            # Reactor pressure affects purge
            ("Reactor Pressure", "Purge Rate", "Reactor pressure affects purge rate"),
            ("Purge (MV)", "Purge Rate", "Purge valve controls purge rate"),
            
            # Agitator affects reactor mixing
            ("Agitator (MV)", "Reactor Temperature", "Agitator affects temperature uniformity"),
            ("Agitator (MV)", "Reactor Pressure", "Agitator affects pressure uniformity"),
            
            # === SEPARATOR SYSTEM ===
            # Separator control loops
            ("Product Sep Level", "Product Sep Underflow", "Sep level affects underflow"),
            ("Product Sep Pressure", "Product Sep Temp", "Sep pressure affects temperature"),
            ("Separator (MV)", "Product Sep Level", "Separator valve controls level"),
            ("Separator (MV)", "Product Sep Underflow", "Separator valve controls underflow"),
            
            # Separator cooling
            ("Product Sep Temp", "Separator Coolant Temp", "Sep temp affects coolant temp"),
            ("Condenser Coolant (MV)", "Separator Coolant Temp", "Condenser valve controls coolant"),
            ("Separator Coolant Temp", "Product Sep Temp", "Coolant temp controls sep temp"),
            
            # === STRIPPER SYSTEM ===
            # Stripper receives separator underflow
            ("Product Sep Underflow", "Stripper Level", "Sep underflow feeds stripper"),
            ("Product Sep Underflow", "Stripper Pressure", "Sep underflow affects stripper pressure"),
            
            # Stripper control loops
            ("Stripper Level", "Stripper Underflow", "Stripper level affects underflow"),
            ("Stripper Pressure", "Stripper Temp", "Stripper pressure affects temperature"),
            ("Stripper (MV)", "Stripper Level", "Stripper valve controls level"),
            ("Stripper (MV)", "Stripper Underflow", "Stripper valve controls underflow"),
            
            # Steam system
            ("Steam (MV)", "Stripper Steam Flow", "Steam valve controls steam flow"),
            ("Stripper Steam Flow", "Stripper Temp", "Steam flow affects stripper temp"),
            ("Stripper Steam Flow", "Stripper Pressure", "Steam flow affects stripper pressure"),
            
            # === COMPRESSOR SYSTEM ===
            # Compressor handles recycle stream
            ("Stripper Underflow", "Compressor Work", "Stripper underflow to compressor"),
            ("Compressor Work", "Recycle Flow", "Compressor creates recycle flow"),
            ("Condenser Coolant (MV)", "Compressor Work", "Condenser cooling affects compressor"),
            
            # === COMPOSITION ANALYZERS ===
            # Reactor composition analyzers
            ("A Feed", "Comp A to Reactor", "A feed affects A composition in reactor"),
            ("D Feed", "Comp D to Reactor", "D feed affects D composition in reactor"),
            ("E Feed", "Comp E to Reactor", "E feed affects E composition in reactor"),
            ("A and C Feed", "Comp A to Reactor", "A&C feed affects A composition in reactor"),
            ("A and C Feed", "Comp C to Reactor", "A&C feed affects C composition in reactor"),
            ("Recycle Flow", "Comp A to Reactor", "Recycle affects A composition in reactor"),
            ("Recycle Flow", "Comp B to Reactor", "Recycle affects B composition in reactor"),
            ("Recycle Flow", "Comp C to Reactor", "Recycle affects C composition in reactor"),
            
            # Reactor mixing affects all compositions
            ("Reactor Temperature", "Comp A to Reactor", "Reactor temp affects A composition"),
            ("Reactor Temperature", "Comp B to Reactor", "Reactor temp affects B composition"),
            ("Reactor Temperature", "Comp C to Reactor", "Reactor temp affects C composition"),
            ("Reactor Temperature", "Comp D to Reactor", "Reactor temp affects D composition"),
            ("Reactor Temperature", "Comp E to Reactor", "Reactor temp affects E composition"),
            ("Reactor Temperature", "Comp F to Reactor", "Reactor temp affects F composition"),
            
            # Purge composition analyzers
            ("Purge Rate", "Comp A in Purge", "Purge rate affects A in purge"),
            ("Purge Rate", "Comp B in Purge", "Purge rate affects B in purge"),
            ("Purge Rate", "Comp C in Purge", "Purge rate affects C in purge"),
            ("Purge Rate", "Comp D in Purge", "Purge rate affects D in purge"),
            ("Purge Rate", "Comp E in Purge", "Purge rate affects E in purge"),
            ("Purge Rate", "Comp F in Purge", "Purge rate affects F in purge"),
            ("Purge Rate", "Comp G in Purge", "Purge rate affects G in purge"),
            ("Purge Rate", "Comp H in Purge", "Purge rate affects H in purge"),
            
            # Product composition analyzers
            ("Product Sep Underflow", "Comp D in Product", "Sep underflow affects D in product"),
            ("Product Sep Underflow", "Comp E in Product", "Sep underflow affects E in product"),
            ("Product Sep Underflow", "Comp F in Product", "Sep underflow affects F in product"),
            ("Product Sep Underflow", "Comp G in Product", "Sep underflow affects G in product"),
            ("Product Sep Underflow", "Comp H in Product", "Sep underflow affects H in product"),
            
            # Stripper affects product composition
            ("Stripper Temp", "Comp G in Product", "Stripper temp affects G in product"),
            ("Stripper Temp", "Comp H in Product", "Stripper temp affects H in product"),
            ("Stripper Underflow", "Comp G in Product", "Stripper underflow affects G in product"),
            ("Stripper Underflow", "Comp H in Product", "Stripper underflow affects H in product"),
            
            # === CROSS-SYSTEM INTERACTIONS ===
            # Reactor conditions affect separator
            ("Reactor Pressure", "Product Sep Pressure", "Reactor pressure affects separator"),
            ("Reactor Temperature", "Product Sep Temp", "Reactor temp affects separator temp"),
            
            # Separator conditions affect stripper
            ("Product Sep Pressure", "Stripper Pressure", "Sep pressure affects stripper"),
            ("Product Sep Temp", "Stripper Temp", "Sep temp affects stripper temp"),
            
            # Recycle loop connections
            ("Stripper Underflow", "Recycle Flow", "Stripper underflow contributes to recycle"),
            ("Compressor Work", "Reactor Pressure", "Compressor work affects reactor pressure"),
            
            # Temperature cascade effects
            ("Reactor Coolant Temp", "Separator Coolant Temp", "Reactor cooling affects separator cooling"),
            ("Separator Coolant Temp", "Compressor Work", "Separator cooling affects compressor work"),
        ]
        
        return relationships
        
    def _build_complex(self):
        """Build the combinatorial complex for TEP"""
        print("Building TEP combinatorial complex...")
        
        # Get all unique components from column names
        unique_components = self.column_names.copy()
        
        # Add components as rank 0 cells (nodes)
        print(f"Adding {len(unique_components)} components as rank 0 cells")
        for component in unique_components:
            self.complex.add_cell([component], rank=0)
        
        # Add specific component relationships as rank 1 cells
        relationships = self._get_specific_component_relationships()
        print(f"Adding {len(relationships)} specific component relationships as rank 1 cells")
        added_count = 0
        
        for source, target, description in relationships:
            try:
                # Check if components exist
                if source in unique_components and target in unique_components:
                    # Add both components as a 1-cell (relation)
                    self.complex.add_cell([source, target], rank=1, name=f"{source}_{target}", description=description)
                    print(f"  Added 1-cell: [{source}, {target}] ({description})")
                    added_count += 1
                else:
                    print(f"  Warning: Could not add 1-cell [{source}, {target}] (component not found)")
            except Exception as e:
                print(f"  Error adding 1-cell [{source}, {target}]: {str(e)}")
        
        print(f"  Successfully added {added_count} component relationships")
        
        # Add subprocess control groups as rank 2 cells (using get_sensor_subsets)
        print(f"Adding subprocess control groups as rank 2 cells")
        subprocess_added_count = 0
        
        for i, components in enumerate(self.plc_components):
            # Convert indices to component names
            valid_components = []
            for comp_idx in components:
                if isinstance(comp_idx, int) and comp_idx < len(unique_components):
                    valid_components.append(unique_components[comp_idx])
                elif isinstance(comp_idx, str) and comp_idx in unique_components:
                    valid_components.append(comp_idx)
            
            if len(valid_components) >= 2:
                # Use the subprocess labels for naming
                subprocess_labels = [
                    'XMV_1', 'XMV_2', 'XMV_3', 'XMV_4', 'XMV_6', 'XMV_7', 'XMV_8', 'XMV_10', 'XMV_11', 'Physical'
                ]
                cell_name = subprocess_labels[i] if i < len(subprocess_labels) else f"Subprocess_{i+1}"
                
                print(f"  Attempting to add {cell_name} with components: {valid_components}")
                try:
                    self.complex.add_cell(valid_components, rank=2, name=cell_name)
                    print(f"  Added rank 2 cell: {cell_name} with {len(valid_components)} components")
                    subprocess_added_count += 1
                except Exception as e:
                    print(f"  Error adding {cell_name}: {str(e)}")
            else:
                print(f"  Warning: Not enough valid components for subprocess {i+1}")
        
        print(f"  Successfully added {subprocess_added_count} subprocess control groups")
        
        print(f"Complex built: {self.complex}")
    
    def get_complex(self):
        """Return the combinatorial complex"""
        return self.complex

# Example usage
def main():
    # Create the TEP complex
    tep_complex = TEPComplex()
    
    # Print complex information
    print("\nComplex information:")
    print(tep_complex.get_complex())

if __name__ == "__main__":
    main() 