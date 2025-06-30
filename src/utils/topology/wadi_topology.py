"""
WADI Topology - Combinatorial Complex representation of WADI system
with detailed component relationships as 1-cells
"""

import numpy as np
import toponetx as tnx
from src.utils.attack_utils import get_attack_indices, get_attack_sds, WADI_SUB_MAP, is_actuator
from collections import defaultdict
from itertools import combinations
import json
import os

class WADIComplex:
    def __init__(self, component_names=None):
        """Initialize the WADI combinatorial complex representation"""
        self.complex = tnx.CombinatorialComplex()
        self.subsystem_map = WADI_SUB_MAP
        self.attacks, self.attack_labels = get_attack_indices("WADI")
        
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
        
        # Build the complex
        self._build_complex(component_names)
        
    def _is_sensor(self, component):
        """Check if a component is a sensor based on its naming pattern"""
        return not is_actuator("WADI", component)
    
    def _is_actuator(self, component):
        """Check if a component is an actuator based on its naming pattern"""
        return is_actuator("WADI", component)
    
    def _get_specific_component_relationships(self):
        """Define specific component relationships for WADI based on process knowledge,
        as described in user documentation."""
        # This list is based on a detailed semantic breakdown of the WADI process.
        relationships = [
            # 1. Raw_Water_Tank (Stage 1)
            # Level transmitter (LT) controls inlet valve (MV) and transfer pump (P)
            ("1_LT_001_PV", "1_MV_001_STATUS", "Tank level controls inlet valve"),
            ("1_LT_001_PV", "1_P_003_STATUS", "Tank level controls transfer pump to stage 2"),
            # Inlet valve and circulation pumps affect inflow (FIT)
            ("1_MV_001_STATUS", "1_FIT_001_PV", "Inlet valve status affects inflow rate"),
            ("1_P_001_STATUS", "1_FIT_001_PV", "Circulation pump P1 affects inflow rate"),
            ("1_P_002_STATUS", "1_FIT_001_PV", "Circulation pump P2 affects inflow rate"),
            # Inflow affects tank level
            ("1_FIT_001_PV", "1_LT_001_PV", "Inflow rate affects tank level"),
            # Low-level alarms (LS) stop pumps
            ("1_LS_001_AL", "1_P_001_STATUS", "Low-level alarm stops circulation pump P1"),
            ("1_LS_001_AL", "1_P_002_STATUS", "Low-level alarm stops circulation pump P2"),
            ("1_LS_002_AL", "1_P_001_STATUS", "Low-level alarm 2 stops circulation pump P1"),
            ("1_LS_002_AL", "1_P_002_STATUS", "Low-level alarm 2 stops circulation pump P2"),
            # Dosing pumps are related to water quality (AIT)
            ("1_P_004_STATUS", "1_AIT_001_PV", "Dosing pump affects conductivity"),
            ("1_P_005_STATUS", "1_AIT_002_PV", "Dosing pump affects turbidity"),
            ("1_P_006_STATUS", "1_AIT_003_PV", "Dosing pump affects pH"),

            # 2. Elevated (Stage 2) & Cross-Subsystem links
            # Transfer pump from S1 affects inflow to S2
            ("1_P_003_STATUS", "2_FIT_001_PV", "S1 transfer pump affects S2 inflow"),
            # S2 inflow affects S2 level
            ("2_FIT_001_PV", "2_LT_001_PV", "S2 inflow rate affects S2 tank level"),
            # S2 tank level gates the transfer pump from S1
            ("2_LT_001_PV", "1_P_003_STATUS", "S2 tank level gates S1 transfer pump"),
            # S2 level transmitters control the main outlet valve to consumers
            ("2_LT_001_PV", "2_MV_001_STATUS", "S2 level gates main consumer valve"),
            ("2_LT_002_PV", "2_MV_001_STATUS", "S2 level backup gates main consumer valve"),
            # Head pressure is related to level
            ("2_LT_001_PV", "2_PIT_001_PV", "Tank level determines head pressure"),
            
            # 3. Booster (Stage 2)
            # Pressure controller loop
            ("2_PIT_002_PV", "2_PIC_003_PV", "Discharge pressure is input to pressure controller"),
            ("2_PIT_003_PV", "2_PIC_003_PV", "Suction pressure is input to pressure controller"),
            ("2_PIC_003_PV", "2_PIC_003_CO", "Pressure controller calculates output"),
            # Controller output affects pump speed and modulating valve
            ("2_PIC_003_CO", "2_P_003_SPEED", "Controller output sets pump speed"),
            ("2_PIC_003_CO", "2_P_004_SPEED", "Controller output sets pump speed"),
            ("2_PIC_003_CO", "2_MCV_007_CO", "Controller output drives booster valve"),
            # Pump speed/status affects pressures
            ("2_P_003_SPEED", "2_PIT_002_PV", "Pump speed affects discharge pressure"),
            ("2_P_004_SPEED", "2_PIT_002_PV", "Pump speed affects discharge pressure"),
            ("2_P_003_STATUS", "2_DPIT_001_PV", "Pump status affects differential pressure"),
            ("2_P_004_STATUS", "2_DPIT_001_PV", "Pump status affects differential pressure"),
            
            # 5. Return (Stage 3) & Cross-Subsystem links
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
            ("3_MV_001_STATUS", "3_P_001_STATUS", "Isolation valve for return pump P1"),
            ("3_MV_002_STATUS", "3_P_002_STATUS", "Isolation valve for return pump P2"),
            ("3_MV_003_STATUS", "3_P_003_STATUS", "Isolation valve for return pump P3"),
            # Return pumps affect return outflow
            ("3_P_001_STATUS", "3_FIT_001_PV", "Return pump P1 affects return outflow"),
            # Raw water tank (S1) level also controls return pumps (interlock)
            ("1_LT_001_PV", "3_P_001_STATUS", "Raw water level interlocks with return pump P1"),
            # Return pumps feed back to raw water tank (S1)
            ("3_P_001_STATUS", "1_FIT_001_PV", "Return pump P1 affects raw water inflow"),
        ]
        
        # Programmatically add relationships for all 6 consumer lines (101 to 601)
        consumer_ids = ["101", "201", "301", "401", "501", "601"]
        # MV_001 is for consumer 101, MV_002 for 201, etc.
        elevated_valves = ["2_MV_001_STATUS", "2_MV_002_STATUS", "2_MV_003_STATUS", "2_MV_004_STATUS", "2_MV_005_STATUS", "2_MV_006_STATUS"]

        for cid, elev_valve in zip(consumer_ids, elevated_valves):
            relationships.extend([
                # Main flow control loop
                (f"2_FIC_{cid}_PV", f"2_FIC_{cid}_CO", "flow measurement affects controller output"),
                (f"2_FIC_{cid}_CO", f"2_MCV_{cid}_CO", "controller output affects modulating valve"),
                (f"2_MCV_{cid}_CO", f"2_FIC_{cid}_PV", "modulating valve affects flow"),
                # Main valve from elevated tank feeds this consumer line
                (elev_valve, f"2_FIC_{cid}_PV", "elevated tank valve affects consumer flow"),
                # Level switches control isolation valve
                (f"2_LS_{cid}_AH", f"2_MV_{cid}_STATUS", "high level trips isolation valve"),
                (f"2_LS_{cid}_AL", f"2_MV_{cid}_STATUS", "low level trips isolation valve"),
                # Level switches also control drain solenoid
                (f"2_LS_{cid}_AH", f"2_SV_{cid}_STATUS", "high level opens drain solenoid"),
                (f"2_LS_{cid}_AL", f"2_SV_{cid}_STATUS", "low level may trigger drain/fill logic"),
                # Drain solenoid affects return flow
                (f"2_SV_{cid}_STATUS", "3_FIT_001_PV", "consumer drain solenoid affects return flow"),
                # Flow is integrated by totalizer
                (f"2_FIC_{cid}_PV", f"2_FQ_{cid}_PV", "flow rate is integrated into total volume"),
            ])
        
        # --- Add Granger Causality Edges ---
        granger_edges_path = os.path.join(os.path.dirname(__file__), 'granger_causality_edges.json')
        if os.path.exists(granger_edges_path):
            print(f"Loading Granger causality edges from {granger_edges_path}", flush=True)
            with open(granger_edges_path, 'r') as f:
                granger_edges = json.load(f)
            
            # Use a set of frozensets for efficient duplicate checking of existing relationships
            existing_edges = {frozenset({r[0], r[1]}) for r in relationships}
            
            added_granger_count = 0
            for source, target, description in granger_edges:
                # Filter out edges involving excluded components
                if source in self.remove_list or target in self.remove_list:
                    continue
                
                # Filter out AIT-related edges to reduce complexity
                if "_AIT_" in source or "_AIT_" in target:
                    continue

                # Filter to only include edges with the lowest p-value
                if "(P-value: 0.0000)" not in description:
                    continue

                # NEW: Only add edges that connect different subsystems
                source_subsystem = self.component_to_subsystem.get(source)
                target_subsystem = self.component_to_subsystem.get(target)
                if source_subsystem is None or target_subsystem is None or source_subsystem == target_subsystem:
                    continue
                
                # Check for duplicates before adding
                edge_frozenset = frozenset({source, target})
                if edge_frozenset not in existing_edges:
                    relationships.append((source, target, description))
                    existing_edges.add(edge_frozenset)
                    added_granger_count += 1
            print(f"Added {added_granger_count} new, unique edges from Granger causality analysis.", flush=True)

        return relationships
        
    def _build_complex(self, component_names=None):
        """Build the combinatorial complex for WADI"""
        print("Building WADI combinatorial complex...", flush=True)
        
        # If no component list is provided, generate it from the subsystem map
        if component_names is None:
            print("No component list provided, building from full topology.", flush=True)
            all_components = []
            for components in self.subsystem_map.values():
                all_components.extend(components)
            
            # Filter components and remove duplicates, preserving order
            unique_components = []
            for comp in all_components:
                if comp not in self.remove_list and comp not in unique_components:
                    unique_components.append(comp)
        else:
            print(f"Building complex with a provided list of {len(component_names)} components.", flush=True)
            unique_components = component_names
        
        # 1. Add components as rank 0 cells (nodes)
        print(f"Adding {len(unique_components)} components as rank 0 cells (nodes)...", flush=True)
        for component in unique_components:
            self.complex.add_cell([component], rank=0)
            
        # 2. Add specific component relationships as rank 1 cells (edges)
        relationships = self._get_specific_component_relationships()
        print(f"Attempting to add {len(relationships)} specific component relationships as rank 1 cells (edges)...", flush=True)
        added_count = 0
        for source, target, description in relationships:
            # Ensure both components exist in the complex before adding an edge
            if source in unique_components and target in unique_components:
                # Use a canonical representation to avoid duplicate edges
                edge_frozenset = frozenset({source, target})
                if edge_frozenset not in self.complex.cells:
                    self.complex.add_cell([source, target], rank=1, name=f"{source}_{target}", description=description)
                    added_count += 1
            # This part is too verbose, let's remove it for cleaner output
            # else:
            #     if source not in unique_components:
            #         # print(f"  Warning: Source component '{source}' not found. Skipping edge [{source}, {target}].")
            #         pass
            #     if target not in unique_components:
            #         # print(f"  Warning: Target component '{target}' not found. Skipping edge [{source}, {target}].")
            #         pass
        
        print(f"Successfully added {added_count} unique component relationships.", flush=True)
            
        # 3. Add subsystems as rank 2 cells (faces)
        print(f"Adding {len(self.subsystem_map)} subsystems as rank 2 cells (faces)...", flush=True)
        subsystem_added_count = 0
        for subsystem, components in self.subsystem_map.items():
            # Filter the components for this subsystem to only those that exist in the complex
            valid_components = [comp for comp in components if comp in unique_components]
            if len(valid_components) >= 2:
                self.complex.add_cell(valid_components, rank=2, name=subsystem)
                subsystem_added_count += 1
            else:
                print(f"  Warning: Not enough valid components for subsystem '{subsystem}' to form a 2-cell.", flush=True)
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