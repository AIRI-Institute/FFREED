#!/usr/bin/env python
import os
from multiprocessing import Pool
from subprocess import run
import glob
import numpy as np
from tempfile import NamedTemporaryFile


class DockingVina:
    def __init__(self, config):
        self.config = config

    def __call__(self, smile):
        affinities = list()
        for i in range(self.config['n_conf']):
            os.environ['OB_RANDOM_SEED'] = str(self.config['seed'] + i)
            affinities.append(DockingVina.docking(smile, **self.config))
        return min(affinities)

    @staticmethod
    def docking(smile, *, vina_program, receptor, box_center,
                box_size, error_val, seed, num_modes, exhaustiveness,
                timeout_dock, timeout_gen3d, **kwargs):

        with NamedTemporaryFile(mode=('r+t')) as f1, NamedTemporaryFile(mode=('r+t')) as f2:
            ligand = f1.name
            docking_file = f2.name
            run_line = "obabel -:{} --gen3D -h -opdbqt -O {}".format(smile, ligand)
            try:
                result = run(run_line.split(), capture_output=True, text=True, timeout=timeout_gen3d, env=os.environ)
            except:
                return error_val

            if "Open Babel Error" in result.stdout or "3D coordinate generation failed" in result.stdout:
                return error_val

            run_line = vina_program
            run_line += " --receptor {} --ligand {} --out {}".format(receptor, ligand, docking_file)
            run_line += " --center_x {} --center_y {} --center_z {}".format(*box_center)
            run_line += " --size_x {} --size_y {} --size_z {}".format(*box_size)
            run_line += " --num_modes {}".format(num_modes)
            run_line += " --exhaustiveness {}".format(exhaustiveness)
            run_line += " --seed {}".format(seed)
            try:
                result = run(run_line.split(), capture_output=True, text=True, timeout=timeout_dock)
            except:
                return error_val

            return DockingVina.parse_output(result.stdout, error_val)
    
    @staticmethod
    def parse_output(result, error_val):
        result_lines = result.split('\n')
        check_result = False
        affinity = error_val

        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            break
        return affinity
