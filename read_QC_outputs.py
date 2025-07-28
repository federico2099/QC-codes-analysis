#################################
# Author  Federico J. Hernandez #
#          Sep 12 2023          #
#################################
#!/usr/bin/env python3

import json
import os
import numpy as np
import warnings
import argparse
import configparser
from abc import ABC, abstractmethod

def set_default_config_values(config, cwd):
    """
    Sets default values for various sections in the config file.

    Parameters:
    - config: A ConfigParser object.
    - cwd: A string representing the current working directory.
    """
    # Default values for the DEFAULT section
    config['DEFAULT'] = {
        'input_dir': cwd,
        'output_dir': cwd,
        'output_file': 'QC_output.dat',
        'num_dir': '1',
        'dir_prefix' : 'dir',
    }
    
    # Default values for the TM section
    config['TM'] = {
        'filename': 'ricc2.out',
        'hess_filename': 'hessian'
    }
    
    # Default values for the Molcas section
    config['Molcas'] = {
        'filename': 'molcas.log',
        'method': 'rasscf'
    }

    # Default values for the Orca section
    config['Orca'] = {
        'method': 'tddft'
    }

    # Default values for the OpenQP section
    config['OpenQP'] = {
        'method': 'mrsf',
        'filename': 'mrsf.log'
    }

def setup_calc(calc_name, calc_type):
    """Return a calculation of the correct subclass"""
    calc_type = calc_type.lower()
    calc_types = {"turbomole": Turbo,
                  "qchem": QChem,
                  "orca": Orca,
                  "molcas": Molcas,
                  "gaussian": G16}
    try:
        return calc_types[calc_type](calc_name)
    except KeyError:
        print(f"Unrecognised program: {calc_type}")
        return None

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

def _write_common_format(array, out_file):
    width = 15
    N = array.shape[0]
    if len(array.shape) == 2:
        row_elements = []
        for i in range(N):
            for j in range(i + 1): # Iterate only over the lower triangular part
                form_element = f"{array[i, j]:.8E}".rjust(width)
                row_elements.append(f"{form_element}")
                if len(row_elements) == 5:
                    out_file.write(' ' + ' '.join(row_elements) + "\n")
                    row_elements = []
        if row_elements:
            out_file.write(' ' + ' '.join(row_elements) + "\n")
    else:
        for i in range(0, N, 5):
            slice = array[i:i+5]
            line = ' ' + ' '.join(f"{num:.8E}".rjust(width) for num in slice)
            out_file.write(line + '\n')

    return None

class CustomError(Exception):
    pass

class Prog(ABC):
    def __init__(self, calc_name_in=None):
        self.calc_name = calc_name_in

    def per_table(self,key,value,at_number=False):
        """
        Retrieve element properties either by symbol or atomic number.

        Parameters:
        key (str or int): The atomic symbol or atomic number of the element.
        at_number (bool): If False, the key is assumed to be an atomic symbol.
                          If True, the key is assumed to be an atomic number.

        Returns:
        dict: A dictionary containing the properties of the requested element.
        """

        # Define the original periodic table dictionary
        per_tab = {
                            'H'  : {'at_num': 1,   'at_mass':  1.007825  },
            'He' : {'at_num': 2,   'at_mass':  4.002603  },
            'Li' : {'at_num': 3,   'at_mass':  7.016004  },
            'Be' : {'at_num': 4,   'at_mass':  9.012182  },
            'B'  : {'at_num': 5,   'at_mass':  11.009305 },
            'C'  : {'at_num': 6,   'at_mass':  12.000000 },
            'N'  : {'at_num': 7,   'at_mass':  14.003074 },
            'O'  : {'at_num': 8,   'at_mass':  15.994915 },
            'F'  : {'at_num': 9,   'at_mass':  18.998403 },
            'Ne' : {'at_num': 10,  'at_mass':  19.992440 },
            'Na' : {'at_num': 11,  'at_mass':  22.989770 },
            'Mg' : {'at_num': 12,  'at_mass':  23.985042 },
            'Al' : {'at_num': 13,  'at_mass':  26.981538 },
            'Si' : {'at_num': 14,  'at_mass':  27.976927 },
            'P'  : {'at_num': 15,  'at_mass':  30.973762 },
            'S'  : {'at_num': 16,  'at_mass':  31.972071 },
            'Cl' : {'at_num': 17,  'at_mass':  34.968853 },
            'Ar' : {'at_num': 18,  'at_mass':  39.962383 },
            'K'  : {'at_num': 19,  'at_mass':  38.963707 },
            'Ca' : {'at_num': 20,  'at_mass':  39.962591 },
            'Sc' : {'at_num': 21,  'at_mass':  44.955910 },
            'Ti' : {'at_num': 22,  'at_mass':  47.947947 },
            'V'  : {'at_num': 23,  'at_mass':  50.943964 },
            'Cr' : {'at_num': 24,  'at_mass':  51.940512 },
            'Mn' : {'at_num': 25,  'at_mass':  54.938050 },
            'Fe' : {'at_num': 26,  'at_mass':  55.934942 },
            'Co' : {'at_num': 27,  'at_mass':  58.933200 },
            'Ni' : {'at_num': 28,  'at_mass':  57.935348 },
            'Cu' : {'at_num': 29,  'at_mass':  62.929601 },
            'Zn' : {'at_num': 30,  'at_mass':  63.929147 },
            'Ga' : {'at_num': 31,  'at_mass':  68.925581 },
            'Ge' : {'at_num': 32,  'at_mass':  73.921178 },
            'As' : {'at_num': 33,  'at_mass':  74.921596 },
            'Se' : {'at_num': 34,  'at_mass':  79.916522 },
            'Br' : {'at_num': 35,  'at_mass':  78.918338 },
            'Kr' : {'at_num': 36,  'at_mass':  83.911507 },
            'Rb' : {'at_num': 37,  'at_mass':  84.911789 },
            'Sr' : {'at_num': 38,  'at_mass':  87.905614 },
            'Y'  : {'at_num': 39,  'at_mass':  88.905848 },
            'Zr' : {'at_num': 40,  'at_mass':  89.904704 },
            'Nb' : {'at_num': 41,  'at_mass':  92.906378 },
            'Mo' : {'at_num': 42,  'at_mass':  97.905408 },
            'Tc' : {'at_num': 43,  'at_mass':  98.907216 },
            'Ru' : {'at_num': 44,  'at_mass':  101.904350},
            'Rh' : {'at_num': 45,  'at_mass':  102.905504},
            'Pd' : {'at_num': 46,  'at_mass':  105.903483},
            'Ag' : {'at_num': 47,  'at_mass':  106.905093},
            'Cd' : {'at_num': 48,  'at_mass':  113.903358},
            'In' : {'at_num': 49,  'at_mass':  114.903878},
            'Sn' : {'at_num': 50,  'at_mass':  118.71    },
            'Sb' : {'at_num': 51,  'at_mass':  120.903818},
            'Te' : {'at_num': 52,  'at_mass':  129.906223},
            'I'  : {'at_num': 53,  'at_mass':  126.904468},
            'Xe' : {'at_num': 54,  'at_mass':  131.904154},
            'Cs' : {'at_num': 55,  'at_mass':  132.905447},
            'Ba' : {'at_num': 56,  'at_mass':  137.905241},
            'La' : {'at_num': 57,  'at_mass':  138.906348},
            'Ce' : {'at_num': 58,  'at_mass':  139.905435},
            'Pr' : {'at_num': 59,  'at_mass':  140.907648},
            'Nd' : {'at_num': 60,  'at_mass':  141.907719},
            'Pm' : {'at_num': 61,  'at_mass':  144.912744},
            'Sm' : {'at_num': 62,  'at_mass':  151.919729},
            'Eu' : {'at_num': 63,  'at_mass':  152.921227},
            'Gd' : {'at_num': 64,  'at_mass':  157.924101},
            'Tb' : {'at_num': 65,  'at_mass':  158.925343},
            'Dy' : {'at_num': 66,  'at_mass':  163.929171},
            'Ho' : {'at_num': 67,  'at_mass':  164.930319},
            'Er' : {'at_num': 68,  'at_mass':  165.930290},
            'Tm' : {'at_num': 69,  'at_mass':  168.934211},
            'Yb' : {'at_num': 70,  'at_mass':  173.938858},
            'Lu' : {'at_num': 71,  'at_mass':  174.940768},
            'Hf' : {'at_num': 72,  'at_mass':  179.946549},
            'Ta' : {'at_num': 73,  'at_mass':  180.947996},
            'W'  : {'at_num': 74,  'at_mass':  183.950933},
            'Re' : {'at_num': 75,  'at_mass':  186.955751},
            'Os' : {'at_num': 76,  'at_mass':  191.961479},
            'Ir' : {'at_num': 77,  'at_mass':  192.962924},
            'Pt' : {'at_num': 78,  'at_mass':  194.964774},
            'Au' : {'at_num': 79,  'at_mass':  196.966552},
            'Hg' : {'at_num': 80,  'at_mass':  201.970626},
            'Tl' : {'at_num': 81,  'at_mass':  204.974412},
            'Pb' : {'at_num': 82,  'at_mass':  207.976636},
            'Bi' : {'at_num': 83,  'at_mass':  208.980383},
            'Po' : {'at_num': 84,  'at_mass':  208.982416},
            'At' : {'at_num': 85,  'at_mass':  209.987131},
            'Rn' : {'at_num': 86,  'at_mass':  222.017570},
            'Fr' : {'at_num': 87,  'at_mass':  223.019731},
            'Ra' : {'at_num': 88,  'at_mass':  226.025403},
            'Ac' : {'at_num': 89,  'at_mass':  227.027747},
            'Th' : {'at_num': 90,  'at_mass':  232.038050},
            'Pa' : {'at_num': 91,  'at_mass':  231.035879},
            'U'  : {'at_num': 92,  'at_mass':  238.050783},
            'Np' : {'at_num': 93,  'at_mass':  237.048167},
            'Pu' : {'at_num': 94,  'at_mass':  244.064198},
            'Am' : {'at_num': 95,  'at_mass':  243.061373},
            'Cm' : {'at_num': 96,  'at_mass':  247.070347},
            'Bk' : {'at_num': 97,  'at_mass':  247.070299},
            'Cf' : {'at_num': 98,  'at_mass':  251.079580},
            'Es' : {'at_num': 99,  'at_mass':  252.082972},
            'Fm' : {'at_num': 100, 'at_mass':  257.095099},
            'Md' : {'at_num': 101, 'at_mass':  258.098425},
            'No' : {'at_num': 102, 'at_mass':  259.101024},
            'Lr' : {'at_num': 103, 'at_mass':  262.109692},
            'Rf' : {'at_num': 104, 'at_mass':  267.      },
            'Db' : {'at_num': 105, 'at_mass':  268.      },
            'Sg' : {'at_num': 106, 'at_mass':  269.      },
            'Bh' : {'at_num': 107, 'at_mass':  270.      },
            'Hs' : {'at_num': 108, 'at_mass':  270.      },
            'Mt' : {'at_num': 109, 'at_mass':  278.      },
            'Ds' : {'at_num': 110, 'at_mass':  281.      },
            'Rg' : {'at_num': 111, 'at_mass':  282.      },
            'Cn' : {'at_num': 112, 'at_mass':  285.      },
            'Nh' : {'at_num': 113, 'at_mass':  286.      },
            'Fl' : {'at_num': 114, 'at_mass':  289.      },
            'Mc' : {'at_num': 115, 'at_mass':  290.      },
            'Lv' : {'at_num': 116, 'at_mass':  293.      },
            'Ts' : {'at_num': 117, 'at_mass':  294.      },
            'Og' : {'at_num': 118,  'at_mass':  294.      },
        }

        # Create a dictionary indexed by atomic numbers
        atnum2props = {
            data['at_num']: {'symbol': symbol, 'at_mass': data['at_mass']}
            for symbol, data in per_tab.items()
        }

        if at_number:
            # Use atomic numbers as keys
            if key in atnum2props:
                return atnum2props[key][value]
            else:
                raise ValueError(f"No element with atomic number {key}")
        else:
            # Use atomic symbols as keys
            if key in per_tab:
                return per_tab[key][value]
            else:
                raise ValueError(f"No element with symbol '{key}'")

    def set_input_dir(self,config):
        if "input_dir" in config.keys():
            base_dir = str(config["input_dir"])

    def set_output_dir(self, config):
        """
        Define the structure of the output directories
        """
        if "output_dir" in config.keys():
            base_dir = str(config["output_dir"])
        output_dir = os.path.join(base_dir, 'parsed_results', f'{self.calc_name}_results')
        return output_dir
    
    def get_fragment_elements(self,ifrag,config):
        fragment_key = f"frag{ifrag}"
        fragment_elements_config = config.get(fragment_key).split()
        fragment_elements = []
        for element in fragment_elements_config:
            if '-' in element:
                start, end = map(int, element.split('-'))
                fragment_elements.extend(range(start, end + 1))
            else:
                fragment_elements.append(int(element))
       
        fragment_elements = [e - 1 for e in fragment_elements]  # Adjusting to 0-based index

        return fragment_elements

    def fragments_pop(self, n_dim, data, config):
        sums = {}
        n_fragments = int(config.get("n_fragments", 0))
        for i in range(1, n_fragments + 1):
            fragment_elements = self.get_fragment_elements(i,config)
            fragment_sums = [0] * n_dim
            for element in fragment_elements:
                for j in range(len(fragment_sums)):
                    fragment_sums[j] += data[element][j]
            sums[f"Frag {i}"] = fragment_sums

        return sums

    def write_FCclasses_interface(self,natoms,ener,coords,grads,hess):
        """
        Write the .fcc input file for a FCCclasses calculation
        """

        file_name = 'FCclasses_input.fcc'
        if os.path.exists(file_name):
            os.remove(file_name)

        coord_lines = [] # Store all lines in a list
        with open(file_name,"a") as out_file:
            head_lines = [
                "INFO\n",
                "FCclasses input file\n\n",
                "GEOM      UNITS=ANGS\n",
                f"   {natoms}\n",
                "Geometry from fromage optimisation in xyz format\n"
            ]
            out_file.writelines(head_lines) 
         
            for atom in coords:
                atom_symbol = atom[0].capitalize()  # Ensure first letter is uppercase
                if len(atom_symbol) > 1:
                    atom_symbol = atom_symbol[0] + atom_symbol[1:].lower()  # Ensure second letter is lowercase
                coord_lines.append(f"{atom_symbol:>2}  {atom[1]:12.8f}  {atom[2]:12.8f}  {atom[3]:12.8f}\n")
            out_file.writelines(coord_lines)
            out_file.write(" \n")
            out_file.write("ENER      UNITS=AU" + "\n")
            out_file.write("  %s" % float(ener) + "\n")
            out_file.write(" \n")
            out_file.write("GRAD      UNITS=AU" + "\n")
            _write_common_format(grads, out_file)
            out_file.write(" \n")
            out_file.write("HESS      UNITS=AU" + "\n")
            _write_common_format(hess, out_file)
            out_file.write(" \n")
            
        return

    @abstractmethod
    def energies(self):
        pass

    @abstractmethod
    def populations(self):
        pass

    @abstractmethod
    def osc_str(self):
        pass

    def character(self):
        pass

    @abstractmethod
    def write_pop_results(self, sums):
        pass

    @abstractmethod
    def FCclasses(self):
        pass

    @abstractmethod
    def scan(self):
        pass

    def write_pop_header(self,config):
        n_fragments = int(config.get("n_fragments", 0))
        natoms = int(config.get("natoms",0))
        out_file = config.get('output_file')  # Default value is 'output.dat'
        with open(out_file, 'a+') as f:
            pop_header = """
        #################################################################################

                                        Population Analysis

        #################################################################################

        Fragments definition:
        Number of fragments = {n_fragments}

        """.format(n_fragments=n_fragments)
            f.write(pop_header)

            total_atoms = 0
            for i in range(1, n_fragments + 1):
                fragment_key = f"frag{i}"
                fragment_elements = self.get_fragment_elements(i,config)
                fragment_elements = [x + 1 for x in fragment_elements]
                frag_atoms = len(fragment_elements)
                total_atoms += len(fragment_elements)
                f.write("        Fragment %s: %s atoms \n" %(i,frag_atoms))
                f.write("        Atoms in Fragment %s: %s \n" %(i,fragment_elements))
            f.write("        Total number of atoms = %s \n" % total_atoms)
            f.write("\n")
            if natoms > total_atoms:
                f.write(
                    "\n FYI: there are %s atoms missing in your fragments" 
                    "definition \n\n" %(natoms-total_atoms)
                )
            elif natoms < total_atoms:
                f.write(
                    "Your fragments definition contains %s more atoms than the" 
                    "total numeber of atoms in your system \n" %(total_atoms-natoms)
                )
                f.write("I'm dying... :-( \n")
                raise CustomError(
                    "Your fragments definition contains more atoms than the total number" 
                    "of atoms in your system"
                )
            return


    def process_directory(self, base_dir, dir_pref, filename, num_dir, method, magnitude, program):

        magnitude_accumulated = []
        if magnitude == 'energy':
            for i in range(1, num_dir + 1):
                directory = os.path.join(base_dir, f"{dir_pref}{i}")
                in_file = os.path.join(directory, filename)
                if os.path.exists(in_file):
                    with open(in_file, 'r') as f:
                        log = f.readlines()
                    if program == 'molcas':
                        magnitude_accumulated.append(read_Molcas_energy(log, method))
                    if program == 'orca':
                        magnitude_accumulated.append(read_Orca_energy(log, method)) # define default method TDDFT
        if magnitude == 'character':
            for i in range(1, num_dir + 1):
                directory = os.path.join(base_dir, f"{dir_pref}{i}")
                in_file = os.path.join(directory, filename)
                if os.path.exists(in_file):
                    with open(in_file, 'r') as f:
                        log = f.readlines()
                    if program == 'molcas':
                        magnitude_accumulated.append(read_Molcas_character(log))
#                    if program == 'orca':
#                        magnitude_accumulated.append(read_Orca_character(log, method))
        if magnitude == 'osc_str':
            for i in range(1, num_dir + 1):
                directory = os.path.join(base_dir, f"{dir_pref}{i}")
                in_file = os.path.join(directory, filename)
                if os.path.exists(in_file):
                    with open(in_file, 'r') as f:
                        log = f.readlines()
                    if program == 'molcas':
                        magnitude_accumulated.append(read_Molcas_osc_str(log))
                    if program == 'orca':
                        magnitude_accumulated.append(read_Orca_osc_str(log))

        return magnitude_accumulated

    def write_energies_to_file(self, energies, output_file):
        header, energy_lines = self.format_energies(energies)
        with open(output_file, 'w') as file:
            file.write(header)
            file.writelines(energy_lines)

    def format_energies(self, energies):
        num_states = len(energies[0])
        header = '# Step    ' + '    '.join([f'E_S{j+1}' for j in range(num_states)]) + '\n'
        energy_lines = [f'  {idx + 1}.    ' + '       '.join(f'{e:.8f}' for e in energy_row) + '\n' for idx, energy_row in enumerate(energies)]
        return header, energy_lines

    def write_character_to_file(self, character, output_file):
        header, character_lines = self.format_character(character)
        with open(output_file, 'w') as file:
            file.write(header)
            file.writelines(character_lines)

    def format_character(self, character):
        num_states = len(character[0])
        header = '# Step    ' + '    '.join([f'Om_S{j+1}' for j in range(num_states)]) + '\n'
        character_lines = [f'  {idx + 1}.    ' + '       '.join(f'{e:.6f}' for e in char_row) + '\n' for idx, char_row in enumerate(character)]
#        character_linies = [f'  {idx + 1}.    ' + '       '.join(f'{e:.8f}' for e in char_row) + '\n' for idx, char_row in enumerate(character)]
        return header, character_lines

    def write_osc_str_to_file(self, osc_str, output_file):
        header, osc_str_lines = self.format_osc_str(osc_str)
        with open(output_file, 'w') as file:
            file.write(header)
            file.writelines(osc_str_lines)

    def format_osc_str(self, osc_str):
        num_states = len(osc_str[0])
        header = '# Step    ' + '    '.join([f'f_S{j+1}' for j in range(num_states)]) + '\n'
        osc_str_lines = [f'  {idx + 1}.    ' + '       '.join(f'{e:.6f}' for e in osc_str_row) + '\n' for idx, osc_str_row in enumerate(osc_str)]
#        character_linies = [f'  {idx + 1}.    ' + '       '.join(f'{e:.8f}' for e in char_row) + '\n' for idx, char_row in enumerate(character)]
        return header, osc_str_lines

class QChem(Prog):
    def __init__(self, calc_name_in):
        super().__init__(calc_name_in)

    def populations(self,config):
        filename = config.get("pop_filename")
        natoms = int(config.get("natoms", 0))
        n_properties = int(3) # CDFT(-CI) calc has 3 interesting properties 

        with open(filename, 'r') as f:
            lines = f.readlines()

        self.write_pop_header(config)
        count = 0
        for i, line in enumerate(lines):
            if "CDFT Becke Populations" in line:
                count += 1
                values = lines[i+3:i+3+natoms]
                data = string2float(values)
                sums = self.fragments_pop(n_properties,data,config)
                self.write_pop_results(str(count),sums,config)

    def write_pop_results(self, count, sums,config):
        out_file = config.get('output_file')
        header = f"""         CDFT Becke Populations per fragment for State {count} 
        -----------------------------------------------------------------------------------
           fragment         Excess Electrons        Population (a.u.)        Net Spin
        """
        body = "\n".join(
            [f"        {frag.ljust(15)} {result[0]:<25} {result[1]:<25} {result[2]:<25}"
             for frag, result in sums.items()]
        )

        with open(out_file, 'a+') as f:
            f.write(f"{header}{body}\n\n")

    def energies(self,config):
        # Implement the logic to calculate or retrieve energies
        pass

    def osc_str(self,config):
        pass

    def character(self,config):
        pass

    def FCclasses(self,config):
        pass

    def scan(self,config):
        # Implement the logic to calculate or retrieve properties from scan
        pass

class Turbo(Prog):
    def __init__(self, calc_name_in):
        super().__init__(calc_name_in)

    def populations(self,config):
        filename = config.get("pop_filename")
        natoms = int(config.get("natoms", 0))
        n_properties = int(1) # It will only read the charges

        with open(filename, 'r') as f:
            lines = f.readlines()

        self.write_pop_header(config)
        count = -1
        for i, line in enumerate(lines):
            if "Summary of Natural Population Analysis:" in line:
                count += 1
                values = lines[i+5:i+5+natoms]
                data = string2float(values)
                sums = self.fragments_pop(n_properties,data,config)
                if count == 1:
                    gs_sums = sums # Save the S0 charges 

                if count > 0:
                    self.write_pop_results(str(count),gs_sums,sums,config)


    def write_pop_results(self, count, gs_sums, sums, config):
        out_file = config.get('output_file')
        header = f"""         Natural Populations per fragment for State {count}
        -----------------------------------------------------------------------------------
            Fragment         Natural Charge          St{count} - S0 Nat. Charge diff
        """
        with open(out_file, 'a+') as f:
            f.write(header)
            for frag, results in sums.items():
                gs_results = gs_sums.get(frag, [0] * len(results))  # Default to zeros
                difference = [result - gs_result for result, gs_result in zip(results, gs_results)]
                f.write(f"{frag.ljust(15)} {results[0]:<25} {difference[0]:<25}\n")
            f.write("\n")

    def energies(self,config):

        filename = config.get("filename")
        out_file = 'TM_ener_data.dat'
        prop = config.get('property')
        
        write_ener = ['energy', 'all']

        energies = []

        with open(filename) as data:
            lines = data.readlines()

        for line in lines:
            if "Final" in line:
                energies.append(float(line.split()[5]))
            elif "Total Energy" in line:
                energies.append(float(line.split()[3]))
            if "Energy:" in line:
                energies.append(float(line.split()[1])+energies[0])

        energies = np.array(energies)

        if prop in write_ener:
            ener_str = '\n'.join(['%12.6f' % x for x in energies])
            ener_data = f"""# Energies  \n{ener_str}"""

            with open(out_file,"w") as out:
                out.write(ener_data)

        return energies

    def gradient(self,config):
        filename = config.get("filename")
        with open(filename) as data:
            lines = data.readlines()

        grad_x = []
        grad_y = []
        grad_z = []

        for line in lines:
            if line.strip():
                if line[0:2] == "dE":
                    nums = [float(i.replace("D", "E")) for i in line.split()[1:]]
                    if line.split()[0] == "dE/dx":
                        grad_x.extend(nums)
                    if line.split()[0] == "dE/dy":
                        grad_y.extend(nums)
                    if line.split()[0] == "dE/dz":
                        grad_z.extend(nums)
        grad = []

        for dx, dy, dz in zip(grad_x, grad_y, grad_z):
            grad.append(dx)
            grad.append(dy)
            grad.append(dz)
        grad = np.array(grad)

        return grad

    def hessian(self,config):
        hess_file = config.get("hess_filename")
        with open(hess_file) as data:
            lines = data.readlines()
        hess_tmp = []
        for line in lines[1:-1]:
            if "$hessian" in line:
                break
            for num in map(float, line.split()[2:]):
                hess_tmp.append(num)
        hess_tmp = np.array(hess_tmp)
        dim = int(np.sqrt(len(hess_tmp)))
        hess = np.zeros((dim,dim))
        cont = -1
        for i in range(dim):
            for j in range(dim):
                cont += 1
                hess[j,i] = hess_tmp[cont]
        return hess

    def coord(self,config,natoms):
        bohr_rad = 0.52917720859
        filename = config.get("filename")
        coord = []

        with open(filename) as data:
            lines = data.readlines()

        last_index = None
        for n in range(len(lines) - 1, -1, -1): # reverse scan 
            if "atomic coordinates" in lines[n]:
                last_index = n + 1 # line after the last occurance of "atomic coordinates"
                break

        if last_index is not None:
            coord_tmp = lines[last_index : last_index + natoms]
            for line in coord_tmp:
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z, a = parts[:4]  # Extract coordinates and atom symbol
                    coord.append([a, float(x) * bohr_rad, float(y) * bohr_rad, float(z) * bohr_rad])

        return coord

    def FCclasses(self,config):
        natoms = int(config.get("natoms"))
        state = int(config.get("target_state"))
        energies = self.energies(config)
        if state < 0:
            warnings.warn(f"The target state for the energy is {state} and it should be between 0 (GS) and {len(energies)-1}",UserWarning)
        coord = self.coord(config,natoms)
        grad = self.gradient(config)
        hessian = self.hessian(config)
#
        self.write_FCclasses_interface(natoms,energies[state],coord,grad,hessian)
        
        return

    def osc_str(self,config):
        filename = config.get("filename")
        out_file = 'TM_spec_data.dat'
        # get energies to print them along the osc str.
        energies = self.energies(config)

        oos = [0.0] # The first row for energies is the GS energy

        with open(filename, 'r') as data:
            for line in data:
                if 'oscillator strength' in line:
                    oos.append(float(line.split()[-1]))
        oos = np.array(oos)

        spec_lines = ["%12.6f %12.6f" % (e, o) for e, o in zip(energies, oos)]
        spec_data = "#  Energies      Osc Str\n" + "\n".join(spec_lines)

        with open(out_file,"w") as out:
            out.write(spec_data)

        return None

    def character(self,config):
        pass

    def scan(self,config):
        # Implement the logic to calculate or retrieve properties from scan
        pass

def read_Molcas_energy(log, method):
    energies = []
    for line in log:
        if "::    RASSCF root number" in line and method == 'rasscf':
            energies.append(float(line.split()[-1]))
        elif "::    CASPT2 Root" in line and method == 'caspt2':
            energies.append(float(line.split()[-1]))
        elif "::    MS-CASPT2 Root" in line and method == 'ms-caspt2':
            energies.append(float(line.split()[-1]))
        elif "::    XMS-CASPT2 Root" in line and method == 'xms-caspt2':
            energies.append(float(line.split()[-1]))
    return energies

def read_Molcas_character(log):
    character = []
    for line in log:
        if "| S0-S" in line:
            character.append(float(line.split()[-1]))
    return character

def read_Molcas_osc_str(log):
    osc_str = []
    for line in log:
        if "| S0-S" in line:
            osc_str.append(float(line.split()[-1]))
    return osc_str

class Molcas(Prog):
    def __init__(self, calc_name_in):
        super().__init__(calc_name_in)

    def populations(self,config):
        filename = config.get("pop_filename")
        natoms = int(config.get("natoms", 0))
        n_properties = int(4)

        with open(filename, 'r') as f:
            lines = f.readlines()

        self.write_pop_header(config)
        count = 0
        for i, line in enumerate(lines):
            if "Lowdin Population Analysis" in line:
                if len(lines[i+3].split()) > 3:
                    count += 1
                    values = lines[i+3:i+3+natoms]
                    data = string2float(values)
                    sums = self.fragments_pop(n_properties, data, config)
                    self.write_pop_results(str(count),sums, config)

    def write_pop_results(self, count, sums):
        output_filename = self.config.get('output_file', 'output.dat')
        header = f"""         Lowdin Population Analysis for state {count}
        ------------------------------------------------------------------------------------------------------------------
        fragment           Charge (e)                h+                       e-                   Del q (State{count} - S0)
        """

        body = "\n".join(
                [f"{frag.ljust(15)} {result[0]:<25} {result[1]:<25} {result[2]:<25} {result[3]:<25}" 
                 for frag, result in sums.items()])
        with open(output_filename, 'a+') as f:
            f.write(header + body + "\n\n")

        return

    def energies(self,config):

        filename = config.get("filename")
        base_dir = config.get("input_dir")
        output_file = 'Molcas_ener_data.dat'
        dir_pref = config.get("dir_prefix")
        num_dir = int(config.get("num_dir"))
        method = config.get("method").lower()

        if num_dir > 1:
            energies = self.process_directory(base_dir, dir_pref, filename, num_dir, method, 'energy', 'molcas')
        else:
            in_file = os.path.join(base_dir, filename)
            if os.path.exists(in_file):
                with open(in_file, 'r') as f:
                    log = f.readlines()
                energies = [read_Molcas_energy(log, method)]

        self.write_energies_to_file(energies, output_file)

        return

    def character(self,config):

        filename = config.get("filename")
        base_dir = config.get("input_dir")
        output_file = 'State_character.dat'
        dir_pref = config.get("dir_prefix")
        num_dir = int(config.get("num_dir"))
        method = 'casscf' # the method does not matter here
    
        if num_dir > 1:
            self.character = process_directory(base_dir, dir_pref, filename, num_dir, method, 'character', 'molcas')
        else:
            in_file = os.path.join(base_dir, filename)
            if os.path.exists(in_file):
                with open(in_file, 'r') as f:
                    log = f.readlines()
                character = [read_Molcas_character(log)]

        self.write_character_to_file(character, output_file)
        return

    def osc_str(self,config):

        filename = config.get("filename")
        base_dir = config.get("input_dir")
        output_file = 'State_osc_str.dat'
        dir_pref = config.get("dir_prefix")
        num_dir = int(config.get("num_dir"))
        method = 'casscf' # the method does not matter here

        if num_dir > 1:
            self.osc_str = process_directory(base_dir, dir_pref, filename, num_dir, method, 'osc_str', 'molcas')
        else:
            in_file = os.path.join(base_dir, filename)
            if os.path.exists(in_file):
                with open(in_file, 'r') as f:
                    log = f.readlines()
                osc_str = [read_Molcas_osc_str(log)]

        self.write_osc_str_to_file(osc_str, output_file)
        return

    def FCclasses(self,config):
        pass

    def scan(self,config):
        # Implement the logic to calculate or retrieve properties from scan
        pass

def read_Orca_energy(log,method):
    disp = 0
    reading = False
    count = 0
    nstates = -1
    energies = np.zeros(1)

    for line in log:
        if 'nroots' in line.lower():
            nstates = int(line.split()[-1])
            energies = np.zeros(nstates + 1)
            break

    for line in log:
        if "TD-DFT/TDA EXCITED STATES" or "TD-DFT EXCITED STATES" in line:
            reading = True
        if count == nstates:
            reading = False
            count = 0
        if "Total Energy       :" in line:
            energies[count] = float(line.split()[3])
        if reading:
            if line.startswith('STATE'):
                count += 1
                energies[count] = float(line.split()[3]) + energies[0]
        if "Dispersion correction" in line and len(line.split()) == 3:
            disp = float(line.split()[2])

    energies += disp
    return energies.tolist()

def read_Orca_osc_str(log):

    oos = [0.0] # The first row for energies is the GS energy

    osc_str_line='absorption spectrum via transition electric dipole moments'
    iline = None
    for i, line in enumerate(log):
        if 'nroots' in line.lower():
            nstates = int(line.split()[-1])
        if osc_str_line in line.lower().strip():
            iline = i            
    if iline is None:
        raise ValueError("The line %s was not found in the output file." % osc_str_line)
    init_line = iline + 5
    for line in log[init_line:init_line+nstates]:
        oos.append(float(line.split()[6]))
#    oos = np.array(oos)

    return oos

class Orca(Prog):
    def __init__(self, calc_name_in):
        super().__init__(calc_name_in)

    def write_pop_results(self,config):
        # Implement the logic to calculate or retrieve energies
        pass

    def populations(self,config):
        # Implement the logic to calculate or retrieve energies
        pass

    def character(self,config):
        pass

    def energies(self,config):

        filename = config.get("filename")
        base_dir = config.get("input_dir")
        output_file = 'Energies.dat'
        dir_pref = config.get("dir_prefix")
        num_dir = int(config.get("num_dir"))
        method = config.get("method").lower()

        if num_dir > 1:
            energies = self.process_directory(base_dir, dir_pref, filename, num_dir, method, 'energy', 'orca')
        else:
            in_file = os.path.join(base_dir, filename)
            if os.path.exists(in_file):
                with open(in_file, 'r') as f:
                    log = f.readlines()
                energies = [read_Orca_energy(log, method)]

        self.write_energies_to_file(energies, output_file)

        return energies

    def osc_str(self,config):
        filename = config.get("filename")
        base_dir = config.get("input_dir")
        output_file = 'State_osc_str.dat'
        dir_pref = config.get("dir_prefix")
        num_dir = int(config.get("num_dir"))
        method = 'tddft'

        if num_dir > 1:
            osc_str = self.process_directory(base_dir, dir_pref, filename, num_dir, method, 'osc_str', 'orca')
        else:
            in_file = os.path.join(base_dir, filename)
            if os.path.exists(in_file):
                with open(in_file, 'r') as f:
                    log = f.readlines()
                osc_str = [read_Orca_osc_str(log)]

        self.write_osc_str_to_file(osc_str, output_file)

        return

    def coord(self,config,natoms):
        filename = config.get("filename")

        with open(filename) as data:
            lines = data.readlines()

        coord = []

        last_index = None
        for n in range(len(lines) - 1, -1, -1): # reverse scan
            if "CARTESIAN COORDINATES (ANGSTROEM)" in lines[n]:
                last_index = n + 2
                break

        if last_index is None:
            raise ValueError(f"Could not find Cartesian coordinates in the file {filename}.")

        coord_lines = lines[last_index : last_index + natoms]
        coord_lines = [x.split() for x in coord_lines]
        atoms = [line[0] for line in coord_lines]
        xyz = np.array([line[1:4] for line in coord_lines], dtype=float)
        for i in range(natoms):
            coord.append([atoms[i], xyz[i, 0], xyz[i, 1], xyz[i, 2]])

        return coord

    def gradient(self,config):
        filename = config.get("filename")
        with open(filename) as data:
            lines = data.readlines()

        grad = []

        for n in range(len(lines) - 1, -1, -1): # reverse scan
            if "CARTESIAN GRADIENT" in lines[n]:
                orig_line = n + 3
                break

        if orig_line is None:
            raise ValueError(f"Could not find Cartesian coordinates in the file {filename}.")

        for line in lines[orig_line:]:
            if len(line.split()) == 6:
                atom_grads = [float(i) for i in line.split()[3:]]
                grad = np.concatenate((grad,atom_grads))
            else:
                break

        grad = np.array(grad)

        return grad

    def hessian(self,config):
        hess_file = config.get("hess_filename")
        with open(hess_file) as data:
            hess_lines = data.read().splitlines()

        hess = None
        for n, line in enumerate(hess_lines):
            if '$hessian' in line:
                nmode = int(hess_lines[n + 1].split()[0])
                nline = (nmode + 1) * (int(nmode / 5) + (nmode % 5 > 0))
                vects = hess_lines[n + 2: n + 2 + nline]
                nmodes = [[] for _ in range(nmode)]
                for m, i in enumerate(vects):
                    row = m % (nmode + 1) - 1
                    if row >= 0:
                        nmodes[row] += [float(j) for j in i.split()[1:]]
                hess = np.array(nmodes).T.reshape((nmode, nmode))

        if hess is None:
            raise ValueError(f"Hessian data not found in the file {in_name}")

        return hess

    def FCclasses(self,config):
        natoms = int(config.get("natoms"))
        state = int(config.get("target_state"))
        energies = self.energies(config)
        if state < 0:
            warnings.warn(f"The target state for the energy is {state} and it should be between 0 (GS) and {len(energies)-1}",UserWarning)
        coord = self.coord(config,natoms)
        grad = self.gradient(config)
        hessian = self.hessian(config)

        self.write_FCclasses_interface(natoms,energies[0][state],coord,grad,hessian)

        return

    def scan(self,config):
        # Implement the logic to calculate or retrieve properties from scan
        pass

class G16(Prog):
    def __init__(self, calc_name_in):
        super().__init__(calc_name_in)

    def coord(self,M):
        """
        This function convert G16 coordintes to list:
        at_number  X Y Z

        """
        coords = []
        symbols = []
        for line in M:
            index, at_num, at_type, x, y, z = line.split()[0:6]
            at_symbol = self.per_table(int(at_num),'symbol',at_number=True)
            coords.append([float(x), float(y), float(z)])
            symbols.append(at_symbol)
        coords = np.array(coords)

        return symbols, coords

    def populations(self,config):
        # Implement the logic to calculate or retrieve energies
        pass

    def energies(self,config):
        # Implement the logic to calculate or retrieve energies
        pass

    def osc_str(self,config):
        pass

    def character(self,config):
        pass

    def write_pop_results(self,config):
        # Implement the logic to calculate or retrieve energies
        pass

    def FCclasses(self,config):
        pass

    def scan(self,config):
        """
        """
        filename = config.get("filename")
        natoms = int(config.get("natoms"))
        out_file = open(config.get('output_file'),"w",1)
        with open(filename) as data:
            log = data.readlines()

        for i, line in enumerate(log):
            if "Input orientation:" in line:
                coord = []
                coords = log[i + 5: i + 5 + natoms]
                at_symbols, xyz = self.coord(coords)
                out_file.write(f"{natoms}\n\n")
                for j in range(natoms):
                    coord_str = "{:>6} {:10.6f} {:10.6f} {:10.6f}".format(
                        at_symbols[j], xyz[j,0], xyz[j,1], xyz[j,2]) + "\n"
                    out_file.write(coord_str)

        return None

def main():
    parser = argparse.ArgumentParser(
        description="""
                    Process data using specified program and property or properties.
                    The way of excecuting this is 
                    python read_QC_outputs.py -i input -s section
                    
                    input: is the initial dictionary as shown below. It can be named as 
                           you want
                    section: DEFAULT, TM, QChem, Molcas, etc.
                    """,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Config file structure:
[DEFAULT]
input_dir = /path/to/default/input
output_dir = /path/to/default/output
output_file = QC_output.dat
num_dirs = 5
dir_prefix = dir-

[TM]
program = turbomole
property = energy   # energy/osc_str/popul/all Default: energy
filename = ricc2.out # Default: ricc2.out
# in the case the property is popul, the number of atoms of the system and
  the number of fragments and the atoms in each fragment must be defined.
  (The same applies for the other programs)
pop_filename = filename
natoms = # of atoms
n_fragments= #of fragments
frag1= 1-10 12 15 16-23
frag2= 11 13 14 24-40   # a whole regne between atom 1 and 10 is defined by the "-" symbol

[QChem]
program = qchem
property = popul # Only popul for a CDFT calc, for now
;... any other QChem-specific configurations ...

[Molcas]
program = molcas
property = popul # Only popul for a WFA calc, for now
Note: 
- The DEFAULT section provides common configurations that apply across sections.
- Each section like TM or QChem represents different setups.
- Configurations in specific sections override those in DEFAULT.

[G16]
program = gaussian
property = scan
natoms = #number of atoms
Note:
- The DEFAULT section provides common configurations that apply across sections.
- Each section like TM or QChem represents different setups.
- Configurations in specific sections override those in DEFAULT.
"""
    )

    parser.add_argument('-i', '--input', default='input', required=True, help="Path to the input configuration file (Default: input)")
    parser.add_argument('-s', '--section', default='DEFAULT', help="Section in the config file to use (Default: DEFAULT)")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    cwd = os.getcwd()

    # Set default values for the config object
    set_default_config_values(config, cwd)

    config.read(args.input)

    section = args.section
    out_file = config[section]['output_file']
    if os.path.exists(out_file):
        os.remove(out_file)
    program = config[section]['program']
    prog_name = program 
    prog = setup_calc(prog_name, program)

    if not prog:
        return

    prop = config[section]['property']
    if prop == 'popul':
        prog.populations(config[section])
    elif prop == 'energy':
        prog.energies(config[section])
    elif prop == 'osc_str':
        prog.osc_str(config[section])
    elif prop == 'character':
        prog.character(config[section])
    elif prop == 'scan':
        prog.scan(config[section])
    elif prop == 'all':
        prog.populations(config[section])
        prog.energies(config[section])
        prog.osc_str(config[section])
    elif prop == 'FCclasses':
        prog.FCclasses(config[section])

if __name__ == "__main__":
    main()

