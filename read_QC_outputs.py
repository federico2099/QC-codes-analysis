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

class Parser(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def read_and_parse_file(self):
        pass
    
    @abstractmethod
    def calculate_sums(self, data):
        pass
    
    @abstractmethod
    def write_results(self, sums):
        pass

class QChem(Parser):
    def read_and_parse_file(self):
        filename = self.config.get("filename")
        natoms = int(self.config.get("natoms", 0))
        
        with open(filename, 'r') as f:
            lines = f.readlines()

        count = 0

        for i, line in enumerate(lines):
            if "CDFT Becke Populations" in line:
                count += 1
                values = lines[i+3:i+3+natoms]
                data = string2float(values)
                sums = self.calculate_sums(data)
                self.write_results(str(count),sums)

    def calculate_sums(self, data):
        sums = {}
        n_fragments = int(self.config.get("n_fragments", 0))
        for i in range(1, n_fragments + 1):
            fragment_key = f"frag{i}"
            fragment_elements = list(map(lambda x: int(x) - 1, self.config.get(fragment_key).split()))
            fragment_sums = [0, 0, 0]
            for element in fragment_elements:
                for j in range(3):
                    fragment_sums[j] += data[element][j]
            sums[f"Frag {i}"] = fragment_sums
        return sums

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

class Molcas(Parser):
    def read_and_parse_file(self):
        filename = self.config.get("filename")
        natoms = int(self.config.get("natoms", 0))

        with open(filename, 'r') as f:
            lines = f.readlines()

        count = 0

        for i, line in enumerate(lines):
            if "Lowdin Population Analysis" in line:
                if len(lines[i+3].split()) > 3:
                    count += 1
                    values = lines[i+3:i+3+natoms]
                    data = string2float(values)
                    sums = self.calculate_sums(data)
                    self.write_results(str(count),sums)

    def calculate_sums(self, data):
        sums = {}
        n_fragments = int(self.config.get("n_fragments", 0))
        for i in range(1, n_fragments + 1):
            fragment_key = f"frag{i}"
            fragment_elements = list(map(lambda x: int(x) - 1, self.config.get(fragment_key).split()))
            fragment_sums = [0, 0, 0, 0]
            for element in fragment_elements:
                for j in range(4):
                    fragment_sums[j] += data[element][j]
            sums[f"Frag {i}"] = fragment_sums
        return sums

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
        elif program_type == 'molcas':
            program = Molcas(config)
        else:
            print("Invalid program type specified in config.")
            exit()

        program.read_and_parse_file()

