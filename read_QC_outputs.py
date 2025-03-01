# Author  Federico J. Hernandez #
#          Sep 12 2023          #
#                               #
#!/usr/bin/env python3
import json
import os
import numpy as np
from abc import ABC, abstractmethod

def read_config(config_file):
    """
    Reads the JSON configuration file.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def string2float(M):
    """
    Converts string arrays to float arrays.
    """
    M = [[float(x) for x in row.split()[2:]] for row in M]
    return M

class CustomError(Exception):
    pass

class Parser(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def read_and_parse_file(self):
        pass
    
    def get_fragment_elements(self,ifrag):
        fragment_key = f"frag{ifrag}"
        fragment_elements_config = self.config.get(fragment_key).split()
        fragment_elements = []
        for element in fragment_elements_config:
            if '-' in element:
                start, end = map(int, element.split('-'))
                fragment_elements.extend(range(start, end + 1))
            else:
                fragment_elements.append(int(element))
       
        fragment_elements = [e - 1 for e in fragment_elements]  # Adjusting to 0-based index

        return fragment_elements

    def fragments_pop(self, n_dim, data):
        sums = {}
        n_fragments = int(self.config.get("n_fragments", 0))
        for i in range(1, n_fragments + 1):
            fragment_elements = self.get_fragment_elements(i)
            fragment_sums = [0] * n_dim
            for element in fragment_elements:
                for j in range(len(fragment_sums)):
                    fragment_sums[j] += data[element][j]
            sums[f"Frag {i}"] = fragment_sums

        return sums

    @abstractmethod
    def write_results(self, sums):
        pass

    def write_header(self):
        n_fragments = int(self.config.get("n_fragments", 0))
        natoms = int(self.config.get("natoms",0))
        output_filename = self.config.get('output_file', 'output.dat')  # Default value is 'output.dat'
        with open(output_filename, 'a+') as f:
            f.write("#################################################################################\n")
            f.write("\n")
            f.write("                             Population Analysis \n")
            f.write("\n")
            f.write("#################################################################################\n")
            f.write("\n")
            f.write("Fragments definition:\n")
            f.write("Number of fragments = %s \n" % n_fragments)
            total_atoms = 0
            for i in range(1, n_fragments + 1):
                fragment_key = f"frag{i}"
#                frag_atoms = len(self.config.get(fragment_key).split())
                fragment_elements = self.get_fragment_elements(i)
                fragment_elements = [x + 1 for x in fragment_elements]
#                fragment_elements = list(map(lambda x: int(x), self.config.get(fragment_key).split()))
                frag_atoms = len(fragment_elements)
                total_atoms += len(fragment_elements)#frag_atoms
                f.write("Fragment %s: %s atoms \n" %(i,frag_atoms))
                f.write("Atoms in Fragment %s: %s \n" %(i,fragment_elements))
            f.write("Total number of atoms = %s \n" % total_atoms)
            f.write("\n")
            if natoms > total_atoms:
                f.write("!!!!!!!\n")
                f.write("FYI: there are %s atoms missing in your fragments definition \n" %(natoms-total_atoms))
                f.write("!!!!!!!\n")
                f.write("\n")
            elif natoms < total_atoms:
                f.write("Your fragments definition contains %s more atoms than the total numeber of atoms in your system \n" %(total_atoms-natoms))
                f.write("I'm dying... :-( \n")
                raise CustomError("Your fragments definition contains more atoms than the total numeber of atoms in your system")
            return

class QChem(Parser):
    def read_and_parse_file(self):
        filename = self.config.get("filename")
        natoms = int(self.config.get("natoms", 0))
        n_properties = int(3) # CDFT(-CI) calc has 3 interesting properties 

        with open(filename, 'r') as f:
            lines = f.readlines()

        self.write_header()
        count = 0
        for i, line in enumerate(lines):
            if "CDFT Becke Populations" in line:
                count += 1
                values = lines[i+3:i+3+natoms]
                data = string2float(values)
                sums = self.fragments_pop(n_properties,data)
                self.write_results(str(count),sums)

    def write_results(self, count, sums):
        output_filename = self.config.get('output_file', 'output.dat')  # Default value is 'output.dat'
        with open(output_filename, 'a+') as f:
            f.write("         CDFT Becke Populations per fragment for State %s \n" % count)
            f.write("-----------------------------------------------------------------------------------\n")
            f.write("fragment         Excess Electrons        Population (a.u.)        Net Spin\n")
            for frag, results in sums.items():
                f.write(f"{frag.ljust(15)} {results[0]:<25} {results[1]:<25} {results[2]:<25} \n")
            f.write("\n")
        return

class Turbomole(Parser):
    def read_and_parse_file(self):
        filename = self.config.get("filename")
        natoms = int(self.config.get("natoms", 0))
        n_properties = int(1) # It will only read the charges

        with open(filename, 'r') as f:
            lines = f.readlines()

        self.write_header()
        count = -1
        for i, line in enumerate(lines):
            if "Summary of Natural Population Analysis:" in line:
                count += 1
                values = lines[i+5:i+5+natoms]
                data = string2float(values)
                sums = self.fragments_pop(n_properties,data)
                if count == 1:
                    gs_sums = sums # Save the S0 charges 

                if count > 0:
                    self.write_results(str(count),gs_sums,sums)

    def write_results(self, count, gs_sums, sums):
        output_filename = self.config.get('output_file', 'output.dat')  # Default value is 'output.dat'
        with open(output_filename, 'a+') as f:
            f.write("         Natural Populations per fragment for State %s \n" % count)
            f.write("-----------------------------------------------------------------------------------\n")
            f.write("fragment         Natural Charge          St%s - S0 Nat. Charge diff\n" % count)
#            for 
            for frag, results in sums.items():
                gs_results = gs_sums.get(frag, [0]*len(results))  # Defaults to a list of zeros if frag is not in GS_sums
                difference = [result - gs_result for result, gs_result in zip(results, gs_results)]
                f.write(f"{frag.ljust(15)} {results[0]:<25} {difference[0]:<25} \n")
            f.write("\n")
        return

class Molcas_wfa(Parser):
    def read_and_parse_file(self):
        filename = self.config.get("filename")
        natoms = int(self.config.get("natoms", 0))
        n_properties = int(4)

        with open(filename, 'r') as f:
            lines = f.readlines()

        self.write_header()
        count = 0
        for i, line in enumerate(lines):
            if "Lowdin Population Analysis" in line:
                if len(lines[i+3].split()) > 3:
                    count += 1
                    values = lines[i+3:i+3+natoms]
                    data = string2float(values)
                    sums = self.fragments_pop(n_properties, data)
                    self.write_results(str(count),sums)

    def write_results(self, count, sums):
        output_filename = self.config.get('output_file', 'output.dat')  # Default value is 'output.dat'
        with open(output_filename, 'a+') as f:
            f.write("         Lowdin Population Analysis for state %s \n" % count)
            f.write("--------------------------------------------------------------------------------------------------------------\n")
            f.write("fragment           Charge (e)                h+                       e-                   Del q (State%s - S0)\n"% count)
            for frag, results in sums.items():
                f.write(f"{frag.ljust(15)} {results[0]:<25} {results[1]:<25} {results[2]:<25} {results[2]:<25}\n")
            f.write("\n")
        return


if __name__ == '__main__':
    config_file = input("Enter the path to the external JSON config file: ")

    if not os.path.exists(config_file):
        print("Config file not found!")
    else:
        config = read_config(config_file)
        program_type = config.get('program')
        output_file = config.get('output_file', 'output.dat')  # Default value is 'output.dat'

        if os.path.exists(output_file):
            os.remove(output_file)
        if program_type == 'qchem':
            program = QChem(config)
        elif program_type == 'molcas_wfa':
            program = Molcas_wfa(config)
        elif program_type == 'turbomole':
            program = Turbomole(config)
        else:
            print("Invalid program type specified in config.")
            exit()

        program.read_and_parse_file()

