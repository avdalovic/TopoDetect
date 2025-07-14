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
            # Raw water comes in through MV001, is metered by FIT001, and fills Tank1 (LT001)
            ("1_MV_001_STATUS", "1_FIT_001_PV", "inlet valve → inflow meter"),
            ("1_FIT_001_PV",   "1_LT_001_PV",  "inflow meter → tank level"),

            
            # When LT001 drops below low‐level, LS_001/LS_002 alarm and shut off pumps & valve
            ("1_LT_001_PV",  "1_LS_001_AL", "tank level → low level alarm 1"),
            ("1_LT_001_PV",  "1_LS_002_AL", "tank level → low level alarm 2"),
            ("1_LS_001_AL", "1_P_001_STATUS", "low level alarm → shut off pump P1"),
            ("1_LS_001_AL", "1_P_002_STATUS", "low level alarm → shut off pump P2"),
            ("1_LS_002_AL", "1_P_001_STATUS", "low level alarm 2 → shut off pump P1"),
            ("1_LS_002_AL", "1_P_002_STATUS", "low level alarm 2 → shut off pump P2"),
            ("1_LS_001_AL",  "1_MV_001_STATUS","alarm1 → shut off inlet valve"),
            ("1_LS_002_AL",  "1_MV_001_STATUS","alarm2 → shut off inlet valve"),


            ("1_P_005_STATUS", "2_MV_001_STATUS", "transfer pump P005 → Stage2 inlet valve"),
            ("1_P_005_STATUS", "2_MV_003_STATUS", "transfer pump P005 → Stage2 inlet valve"),
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
            ("3_MV_002_STATUS", "1_FIT_001_PV", "Backwash valve MV002 → Stage1 inflow meter FIT001")
        ]
        
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