"""
SWAT Topology - Combinatorial Complex representation of SWAT system
with detailed component relationships as 1-cells
"""

import numpy as np
import toponetx as tnx
from src.utils.attack_utils import get_attack_indices, get_sensor_subsets, SWAT_SUB_MAP, is_actuator

class SWATComplex:
    def __init__(self):
        """Initialize the SWAT combinatorial complex representation"""
        self.complex = tnx.CombinatorialComplex()
        self.subsystem_map = SWAT_SUB_MAP
        self.attacks, self.attack_labels = get_attack_indices("SWAT")
        self.plc_idxs, self.plc_components = get_sensor_subsets("SWAT", by_plc=True)
        self.process_idxs, _ = get_sensor_subsets("SWAT", by_plc=False)
        
        # Build the complex
        self._build_complex()
        
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
        
        # ONLY add specific component relationships as rank 1 cells
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
        
        # # Add subsystems as rank 3 cells
        # print(f"Adding subsystems as rank 3 cells")
        # subsystem_added_count = 0
        
        # for subsystem, components in self.subsystem_map.items():
        #     # Filter out components that don't exist in our list
        #     valid_components = [comp for comp in components if comp in unique_components]
        #     if len(valid_components) >= 2:  # Need at least 2 components to form a cell
        #         self.complex.add_cell(valid_components, rank=3, name=subsystem)
        #         print(f"  Added rank 3 cell: {subsystem} with {len(valid_components)} components")
        #         subsystem_added_count += 1
        #     else:
        #         print(f"  Warning: Not enough valid components for subsystem {subsystem}")
        
        # print(f"  Successfully added {subsystem_added_count} subsystems")
        
        print(f"Complex built: {self.complex}")
    
    def get_complex(self):
        """Return the combinatorial complex"""
        return self.complex
    
    

# Example usage
def main():
    # Create the SWAT complex
    swat_complex = SWATComplex()
    
    # Print complex information
    print("\nComplex information:")
    print(swat_complex.get_complex())

if __name__ == "__main__":
    main()