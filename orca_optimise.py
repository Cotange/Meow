import os
import sys
import shutil
import subprocess
import time
import re
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# Как использовать скрипт так, чтобы он работал независимо от окончания сессии на кластере:
# nohup python -u orca_optimise.py /путь/к/папке/с_системами &

# ------------------------------------------------------------
# Настройки
# ------------------------------------------------------------
# Папка с шаблонами находится рядом со скриптом (в той же директории)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")   # шаблоны и start.sh.template

# ------------------------------------------------------------
# Логирование
# ------------------------------------------------------------
log = logging.getLogger()
log.setLevel(logging.INFO)

# Консольный обработчик
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

# Файловый обработчик будет добавлен после определения корневой директории
file_handler = None

# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------

def pdb_to_inp(pdb_file):

    try:
        atoms = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    parts = line.split()
                    if len(parts) >= 8:
                        atom_type = parts[2]
                        element = atom_type[0]
                        x = float(parts[5])
                        y = float(parts[6])
                        z = float(parts[7])
                        atoms.append((element, x, y, z))

        if not atoms:
            raise ValueError(f"Not a single atom has been found in {pdb_file}")

        element0, x0, y0, z0 = atoms[0]
        coord_block = f"{element0} {x0} {y0} {z0}"
        for element, x, y, z in atoms[1:]:
            coord_block += f"\n{element} {x} {y} {z}"

        template_path = os.path.join(INPUT_DIR, "opt.inp")
        with open(template_path, 'r') as f:
            template = f.read()

        if '[COORDINATES]' not in template:
            raise ValueError(f"Template {template_path} does not contain [COORDINATES] marker")

        result = template.replace('[COORDINATES]', coord_block)

        inp_file = os.path.splitext(pdb_file)[0] + '.inp'
        with open(inp_file, 'w') as f:
            f.write(result)

    except FileNotFoundError:
        logging.warning(f"File '{pdb_file}' has not been found.")

def ensure_unix_format(filepath):
    """Переводит концы строк в Unix-формат."""
    with open(filepath, 'rb') as f:
        content = f.read().replace(b'\r\n', b'\n')
    with open(filepath, 'wb') as f:
        f.write(content)

def prepare_start_script(dest, inp_name, process):
    """
    Создаёт start.sh в папке dest, подставляя имя входного файла inp_name
    в шаблон start.sh.template из INPUT_DIR.
    """
    template_path = os.path.join(INPUT_DIR, "start.sh.template")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template start.sh.template not found in {INPUT_DIR}")
    with open(template_path, 'r') as f:
        content = f.read()
    if '__INP__' not in content:
        raise ValueError("Template does not contain __INP__ placeholder")
    new_content = content.replace('__INP__', inp_name)
    basename = os.path.splitext(inp_name)[0]
    new_content = new_content.replace('orca.out', f"{basename}.out")
    new_content = new_content.replace('process', f"{process}_{basename}")
    start_script_path = os.path.join(dest, f"{basename}.sh")
    with open(start_script_path, 'w') as f:
        f.write(new_content)
    ensure_unix_format(start_script_path)
    os.chmod(start_script_path, 0o755)

def read_xyz_coordinates(xyz_path):
    """
    Читает .xyz файл и возвращает строку с координатами (без первых двух строк).
    """
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    # Пропускаем строку с числом атомов и комментарий
    return ''.join(lines[2:])

def generate_input_file(template_path, coords_text, output_path):
    """
    Создаёт входной файл ORCA, заменяя в шаблоне маркер [COORDINATES]
    на блок реальных координат.
    """
    with open(template_path, 'r') as f:
        content = f.read()
    if '[COORDINATES]' not in content:
        raise ValueError(f"Template {template_path} does not contain [COORDINATES] marker")
    new_content = content.replace('[COORDINATES]', coords_text)
    with open(output_path, 'w') as f:
        f.write(new_content)
    ensure_unix_format(output_path)

def is_calculation_done(folder, out_filename):
    """
    Проверяет, успешно ли завершён расчёт в папке.
    Критерий: наличие файла out_filename и фразы нормального завершения.
    """
    out_file = os.path.join(folder, out_filename)
    if not os.path.exists(out_file):
        return False
    with open(out_file, 'r', errors='ignore') as f:
        content = f.read()
    return "****ORCA TERMINATED NORMALLY****" in content

def submit_job(folder, basename):
    """Отправляет задание SLURM в папке folder и возвращает job ID."""
    result = subprocess.run(
        ["sbatch", f"{basename}.sh"],
        cwd=folder,
        capture_output=True,
        text=True
    )
    job_id = re.search(r"\d+", result.stdout).group()
    logging.info(f"Submitted job {job_id} in {folder}")
    return job_id

def wait_for_job(job_id):
    """Ожидает завершения задания с указанным job ID."""
    while True:
        result = subprocess.run(["squeue", "-j", job_id],
                                capture_output=True, text=True)
        if job_id not in result.stdout:
            logging.info(f"Finished job {job_id}")
            break
        time.sleep(20)

def check_success(folder, out_filename):
    """Проверяет наличие нормального завершения в указанном выходном файле."""
    logging.info(f"Checking success: {folder}")
    out_file = os.path.join(folder, out_filename)
    if not os.path.exists(out_file):
        raise RuntimeError(f"Output file missing: {out_file}")
    with open(out_file, 'r', errors='ignore') as f:
        content = f.read()
    if "****ORCA TERMINATED NORMALLY****" not in content:
        raise RuntimeError(f"Calculation NOT finished successfully in {folder}")
    logging.info(f"Success: {folder}")

def append_if_not_found(filename, search_string, line_to_add):
    """
    Проверяет, содержится ли search_string в файле filename.
    Если нет, добавляет line_to_add в конец файла.
    """
    
    pattern = re.compile(rf'\b{re.escape(word)}\b')
    found = False
    
    with open(filename, 'a+') as f:
        # Перемещаем указатель в начало, чтобы прочитать существующее содержимое
        f.seek(0)
        for line in f:
            if search_string in line:
                found = True
                break

        if not found:
            # Добавляем строку (с переводом строки, чтобы не склеивалось)
            f.write(line_to_add + '\n')


# ------------------------------------------------------------
# Функция для извлечения списка атомов
# ------------------------------------------------------------

class Atom:
    """
    This is the Atom class. All atom objects are expected to contain the usual parameters
    written in a PDB file: atom number, atom name, residue name, residue number and XYZ
    coordinates.
    """
    def __init__(self, atom_number, atom_name, residue_name, residue_number, x, y, z):
        self.atom_number = atom_number
        self.atom_name = atom_name
        self.residue_name = residue_name
        self.residue_number = residue_number
        self.x = x
        self.y = y
        self.z = z

def read_in_GO(pdb_file):
    """Reads in a GO layer where C atom name is "CX", residue name is 'GGG'
    Expected carboyl residue name: C1A; Expected epoxy residue name: E1A; Expected hydroxyl residue name: H1A;"""
    with open(pdb_file, "r") as f:
        filedata = f.read()
        filedata = filedata.replace("C   GRA X", "CX  GGG  ")
        content = filedata.splitlines()
        atom_lines = [x.split() for x in content if (('ATOM' in str(x)) and (('C1A' in str(x)) or ('E1A' in str(x)) or ('H1A' in str(x)) or ('GGG' in str(x))))]
        atoms = [Atom(int(str(atom_lines[x][1])), str(atom_lines[x][2]), str(atom_lines[x][3]), int(str(atom_lines[x][4])), float(str(atom_lines[x][5])), float(str(atom_lines[x][6])), float(str(atom_lines[x][7]))) for x in range(len(atom_lines))] 
    
    return atoms


# ------------------------------------------------------------
# Функция для извлечения долей элементов
# ------------------------------------------------------------

def extract_mass_fraction(pdb_file):

    atom_list = read_in_GO(pdb_file)
    elements = {
        'C': [12, 0, 0, 0, 0],
        'O': [16, 0, 0, 0, 0],
        'H': [1, 0, 0, 0, 0],
        #'N': [14, 0, 0, 0, 0],
        'Total': ['-', 0, 0, 0, 0]
    }
    
    for atom in atom_list:
        сurrent_element = atom.atom_name[0]
        element_mass = elements[сurrent_element][0]
        elements[сurrent_element][1] += 1
        elements[сurrent_element][2] += element_mass

    elements['Total'][1] = sum([elements[element][1] for element in elements])
    elements['Total'][2] = sum([elements[element][2] for element in elements])
    
    for element in elements:
        elements[element][3] = round(elements[element][1] / elements['Total'][1] * 100, 2)
        elements[element][4] = round(elements[element][2] /elements['Total'][2] * 100, 2)
    
    import pandas as pd
    
    df = pd.DataFrame.from_dict(elements, orient='index')
    df = df.reset_index()
    df = df.to_string(index=False, header=False)
    
    return df


# ------------------------------------------------------------
# Функция для извлечения долей групп
# ------------------------------------------------------------

def extract_functional_group_fraction(pdb_file):
    
    atom_list = read_in_GO(pdb_file)
    functional_groups = {
        'Graphene_C': [12, 1, 0, 0, 0, 0, 0],
        'Graphene_H': [1, 1, 0, 0, 0, 0, 0],
        'O': [16, 1, 0, 0, 0, 0, 0],
        'OH': [17, 2, 0, 0, 0, 0, 0],
        'COOH': [45, 4, 0, 0, 0, 0, 0],
        'Total Graphene': ['-', '-', '-', 0, 0, 0, 0],
        'Total Functional groups': ['-', '-', '-', 0, 0, 0, 0],
        'Total': ['-', '-', '-', 0, 0, 0, 0]
    }

    for atom in atom_list:
        сurrent_group = atom.residue_name
        сurrent_element = atom.atom_name[0]
        if сurrent_group == 'GGG':
            if сurrent_element == 'C':
                functional_groups['Graphene_C'][2] += 1
            if сurrent_element == 'H':
                functional_groups['Graphene_H'][2] += 1
        elif сurrent_group == 'H1A':
            functional_groups['OH'][2] += 1/3
            functional_groups['Graphene_C'][2] += 1/3
        elif сurrent_group == 'E1A':
            functional_groups['O'][2] += 1/3
            functional_groups['Graphene_C'][2] += 2/3
        elif сurrent_group == 'C1A':
            functional_groups['COOH'][2] += 1/5
            functional_groups['Graphene_C'][2] += 1/5
        else:
            print("Unknown functional group")
            raise ValueError
    
    for group in list(functional_groups.keys())[:-3]:
        functional_groups[group][2] = round(functional_groups[group][2])
        functional_groups[group][3] = functional_groups[group][1] * functional_groups[group][2]
        functional_groups[group][4] = functional_groups[group][0] * functional_groups[group][2]

    functional_groups['Total Graphene'][3] = sum([functional_groups[group][3] for group in list(functional_groups.keys())[:2]])
    functional_groups['Total Graphene'][4] = sum([functional_groups[group][4] for group in list(functional_groups.keys())[:2]])

    functional_groups['Total Functional groups'][3] = sum([functional_groups[group][3] for group in list(functional_groups.keys())[2:-3]])
    functional_groups['Total Functional groups'][4] = sum([functional_groups[group][4] for group in list(functional_groups.keys())[2:-3]])

    functional_groups['Total'][3] = sum([functional_groups[group][3] for group in list(functional_groups.keys())[-3:-1]])
    functional_groups['Total'][4] = sum([functional_groups[group][4] for group in list(functional_groups.keys())[-3:-1]])
        
    for group in functional_groups:    
        functional_groups[group][5] = round(functional_groups[group][3] / functional_groups['Total'][3] * 100, 2)
        functional_groups[group][6] = round(functional_groups[group][4] / functional_groups['Total'][4] * 100, 2)
            
    import pandas as pd
    
    df = pd.DataFrame.from_dict(functional_groups, orient='index')
    df = df.reset_index()
    df = df.to_string(index=False, header=False)
    
    return df


# ------------------------------------------------------------
# Функция записи долей в общий файл
# ------------------------------------------------------------
    
def append_fractions(pdb_file):
    """
    Добавляет строку с gap в файл HUMO-LOMO_gaps.txt в корневой директории.
    """
    fraction_file = os.path.join(root_dir, "Fractions.txt")
    basename = os.path.splitext(os.path.basename(pdb_file))[0]
    mass_fraction_value = extract_mass_fraction(pdb_file)
    group_fraction_value = extract_functional_group_fraction(pdb_file)
    # Если файла нет, создаём с заголовком
    if not os.path.exists(fraction_file):
        with open(fraction_file, 'w') as f:
            f.write("Structure's name\n")
            f.write("\nMass of atom (a. m. u.)\tNumber of atoms\tMass of atoms (a. m. u.)\tAtom fraction (%)\tMass fraction (%)\n")
            f.write("\nMass of group (a. m. u.)\tNumber of atoms per group\tNumber of groups\tNumber of group's atoms\tMass of group's atoms(a. m. u.)\tAtom fraction of group (%)\tMass fraction of group (%)\n")
    
    pattern = re.compile(rf'\b{re.escape(basename)}\b')
    found = False
    
    with open(fraction_file, 'a+') as f:
        # Перемещаем указатель в начало, чтобы прочитать существующее содержимое
        f.seek(0)
        
        for line in f:
            if pattern.search(line):
                found = True
                break

        if not found:
            # Добавляем строку 
            f.write(f"\n{basename}\n\n{mass_fraction_value}\n")
            f.write(f"\n{group_fraction_value}\n")
            log.info(f"Fractions for {basename} written to {fraction_file}")
        else:
            log.info(f"Fractions for {basename} already written to {fraction_file}")


# ------------------------------------------------------------
# Функция извлечения ИК-спектра из выходного файла ORCA
# ------------------------------------------------------------

def extract_IR_spectrum(out_file):
    """
    Извлекает ИК-спектр из выходного файла ORCA.
    Возвращает cписок кортежей (частота, интенсивность) для ИК-спектра или None, если не удалось найти.
    """
    try:
        with open(out_file, 'r') as f:
            content = f.read()
        pattern = r"(\d+):\s+(\d+\.\d*)\s+(\d+\.\d*)\s+(\d+\.\d*)\s+(\d+\.\d*)\s+(\(\s?(-?\d+\.\d*)\s+(-?\d+\.\d*)\s+(-?\d+\.\d*)\))"
        matches = re.findall(pattern, content)
        
        IR_spectrum = []
        for match in matches:
            IR_spectrum.append((float(match[1]), float(match[3])))
        IR_spectrum = pd.DataFrame(IR_spectrum)
        IR_spectrum = IR_spectrum.to_string(index=False, header=False)
        
        return IR_spectrum
    
    except Exception as e:
        log.warning(f"Error reading {out_file}: {e}")
        return None


# ------------------------------------------------------------
# Функция записи ИК-спекртра в общий файл (сразу после расчёта)
# ------------------------------------------------------------

def append_IR_spectrum(basename, IR_spectrum_value):
    """
    Добавляет строку с ИК-спектром в файл IR_spectrum.txt в корневой директории.
    """
    IR_spectrum_file = os.path.join(root_dir, "IR_spectrum.txt")
    # Если файла нет, создаём с заголовком
    if not os.path.exists(IR_spectrum_file):
        with open(IR_spectrum_file, 'w') as f:
            f.write("Structure's name\nFreq (cm-1)\tInt (km/mol)\n")
    
    pattern = re.compile(rf'\b{re.escape(basename)}\b')
    found = False
    
    with open(IR_spectrum_file, 'a+') as f:
        # Перемещаем указатель в начало, чтобы прочитать существующее содержимое
        f.seek(0)
        for line in f:
            if pattern.search(line):
                found = True
                break

        if not found:
            # Добавляем строку 
            f.write(f"\n{basename}\n{IR_spectrum_value}\n")
            log.info(f"IR-spectrum for {basename} written to {IR_spectrum_file}")
        else:
            log.info(f"IR-spectrum for {basename} already written to {IR_spectrum_file}") 

            
# ------------------------------------------------------------
# Функция извлечения оптического спектра из выходного файла ORCA
# ------------------------------------------------------------

def extract_optical_spectrum(out_file):
    """
    Извлекает оптический спектр из выходного файла ORCA.
    Возвращает cписок кортежей (частота, интенсивность) для ИК-спектра или None, если не удалось найти.
    """
    try:
        with open(out_file, 'r') as f:
            content = f.read()
        pattern = r"(\d+\.\d*)\s+(\d+\.\d*)\s+(\d+\.\d*)\s+(\d+\.\d*)\s+(\d+\.\d*)\s+(-?\d+\.\d*)\s+(-?\d+\.\d*)\s+(-?\d+\.\d*)"
        matches = re.findall(pattern, content)
        count = len(matches) // 2
    
        abs_optical_spectrum = []
        for match in matches[:count]:
            abs_optical_spectrum.append((float(match[1]), float(match[3])))
        abs_optical_spectrum = pd.DataFrame(abs_optical_spectrum)
        abs_optical_spectrum = abs_optical_spectrum.to_string(index=False, header=False)

        return abs_optical_spectrum
    
    except Exception as e:
        log.warning(f"Error reading {out_file}: {e}")
        return None


# ------------------------------------------------------------
# Функция записи оптического спекртра в общий файл (сразу после расчёта)
# ------------------------------------------------------------

def append_optical_spectrum(basename, optical_spectrum_value):
    """
    Добавляет строку с оптическим спектром в файл Optical_spectrum.txt в корневой директории.
    """
    optical_spectrum_file = os.path.join(root_dir, "Optical_spectrum.txt")
    # Если файла нет, создаём с заголовком
    if not os.path.exists(optical_spectrum_file):
        with open(optical_spectrum_file, 'w') as f:
            f.write("Structure's name\nFreq (cm-1)\tFosc (D2/P2)\n")
    
    pattern = re.compile(rf'\b{re.escape(basename)}\b')
    found = False
    
    with open(optical_spectrum_file, 'a+') as f:
        # Перемещаем указатель в начало, чтобы прочитать существующее содержимое
        f.seek(0)
        for line in f:
            if pattern.search(line):
                found = True
                break

        if not found:
            # Добавляем строку 
            f.write(f"\n{basename}\n{optical_spectrum_value}\n")
            log.info(f"Optical spectrum for {basename} written to {optical_spectrum_file}")
        else:
            log.info(f"Optical spectrum for {basename} already written to {optical_spectrum_file}") 

            
#------------------------------------------------------------
# Функция извлечения HOMO-LUMO gap из выходного файла ORCA
# ------------------------------------------------------------

def extract_gap(out_file):
    """
    Извлекает HOMO-LUMO gap (в eV) из выходного файла ORCA.
    Возвращает gap (LUMO - HOMO) или None, если не удалось найти.
    """
    try:
        with open(out_file, 'r') as f:
            content = f.read()
        pattern = r"(\d+)\s+(2.0000)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(\d+)\s+(0.0000)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
        matches = re.findall(pattern, content)
        
        homo = float(matches[-1][3])
        lumo = float(matches[-1][7])
        gap = round(lumo-homo, 4)
        
        return gap
    
    except Exception as e:
        log.warning(f"Error reading {out_file}: {e}")
        return None


# ------------------------------------------------------------
# Функция записи HOMO-LUMO gap в общий файл (сразу после расчёта)
# ------------------------------------------------------------

def append_gap(basename, gap_value):
    """
    Добавляет строку с gap в файл HUMO-LOMO_gaps.txt в корневой директории.
    """
    gap_file = os.path.join(root_dir, "HUMO-LOMO_gaps.txt")
    # Если файла нет, создаём с заголовком
    if not os.path.exists(gap_file):
        with open(gap_file, 'w') as f:
            f.write("Structure's name\tGap (eV)\n")
    
    pattern = re.compile(rf'\b{re.escape(basename)}\b')
    found = False
    
    with open(gap_file, 'a+') as f:
        # Перемещаем указатель в начало, чтобы прочитать существующее содержимое
        f.seek(0)
        
        for line in f:
            if pattern.search(line):
                found = True
                break

        if not found:
            # Добавляем строку 
            f.write(f"\n{basename}\t{gap_value}\n")
            log.info(f"Gap for {basename} written to {gap_file}")
        else:
            log.info(f"Gap for {basename} already written to {gap_file}")
    

# --------------------------------------------------------
# Этап 1: грубая оптимизация
# --------------------------------------------------------

def opt(system_path, basename, base_inp):
    opt_dir = os.path.join(system_path, "Vacuum", "opt")
    os.makedirs(opt_dir, exist_ok=True)

    inp_name_opt = f"{basename}.inp"
    out_name_opt = f"{basename}.out"

    if not is_calculation_done(opt_dir, out_name_opt):
        logging.info(f"{basename}: starting opt in {opt_dir}")
        # Копируем исходный .inp (без изменений) и готовим start.sh
        shutil.copy(base_inp, os.path.join(opt_dir, inp_name_opt))
        prepare_start_script(opt_dir, inp_name_opt, 'opt')
        job_id = submit_job(opt_dir, basename)
        wait_for_job(job_id)
        check_success(opt_dir, out_name_opt)
        logging.info(f"{basename}: opt completed")
    else:
        logging.info(f"{basename}: opt already done")

    try:
        os.remove(base_inp)
        logging.info(f"{basename}: removed original input file {base_inp}")
    except OSError as e:
        logging.warning(f"{basename}: could not remove original input file: {e}")


# --------------------------------------------------------
# Этап 2: точная оптимизация в вакууме и воде
# --------------------------------------------------------

def opt_acc(system_path, basename):
    opt_dir = os.path.join(system_path, "Vacuum", "opt")
    opt_xyz = os.path.join(opt_dir, f"{basename}.xyz")
    if not os.path.exists(opt_xyz):
        raise RuntimeError(f"{basename}: {basename}.xyz not found in {opt_dir}")

    opt_acc_dir = os.path.join(system_path, "Vacuum", "opt_acc")
    os.makedirs(opt_acc_dir, exist_ok=True)

    inp_name_acc = f"{basename}.inp"
    out_name_acc = f"{basename}.out"

    if not is_calculation_done(opt_acc_dir, out_name_acc):
        logging.info(f"{basename}: starting opt_acc (vacuum) in {opt_acc_dir}")
        coords = read_xyz_coordinates(opt_xyz)
        template = os.path.join(INPUT_DIR, "opt_acc.inp")
        inp_path = os.path.join(opt_acc_dir, inp_name_acc)
        generate_input_file(template, coords, inp_path)
        prepare_start_script(opt_acc_dir, inp_name_acc, 'opt_acc')
        job_id = submit_job(opt_acc_dir, basename)
        wait_for_job(job_id)
        check_success(opt_acc_dir, out_name_acc)
        logging.info(f"{basename}: opt_acc (vacuum) completed")
    else:
        logging.info(f"{basename}: opt_acc (vacuum) already done")

    out_file = os.path.join(opt_acc_dir, out_name_acc)
    gap_value = extract_gap(out_file)
    append_gap(basename, gap_value)


def H2O_opt_acc(system_path, basename):
    opt_dir = os.path.join(system_path, "Vacuum", "opt")
    opt_xyz = os.path.join(opt_dir, f"{basename}.xyz")
    if not os.path.exists(opt_xyz):
        raise RuntimeError(f"{basename}: {basename}.xyz not found in {opt_dir}")

    opt_acc_water_dir = os.path.join(system_path, "H2O", "opt_acc")
    os.makedirs(opt_acc_water_dir, exist_ok=True)

    inp_name_acc_water = f"H2O_{basename}.inp"
    out_name_acc_water = f"H2O_{basename}.out"

    if not is_calculation_done(opt_acc_water_dir, out_name_acc_water):
        logging.info(f"{basename}: starting opt_acc (water) in {opt_acc_water_dir}")
        coords = read_xyz_coordinates(opt_xyz)
        template = os.path.join(INPUT_DIR, "H2O_opt_acc.inp")
        inp_path = os.path.join(opt_acc_water_dir, inp_name_acc_water)
        generate_input_file(template, coords, inp_path)
        prepare_start_script(opt_acc_water_dir, inp_name_acc_water, 'opt_acc')
        job_id = submit_job(opt_acc_water_dir, 'H2O_' + basename)
        wait_for_job(job_id)
        check_success(opt_acc_water_dir, out_name_acc_water)
        logging.info(f"{basename}: opt_acc (water) completed")
    else:
        logging.info(f"{basename}: opt_acc (water) already done")

    out_file = os.path.join(opt_acc_water_dir, out_name_acc_water)
    gap_value = extract_gap(out_file)
    append_gap(f"H2O_{basename}", gap_value)


# --------------------------------------------------------
# Этап 3.1: расчёт частот ИК-спектра в вакууме и воде
# --------------------------------------------------------

def freq(system_path, basename):
    opt_acc_dir = os.path.join(system_path, "Vacuum", "opt_acc")
    opt_acc_xyz = os.path.join(opt_acc_dir, f"{basename}.xyz") 
    if not os.path.exists(opt_acc_xyz):
        raise RuntimeError(f"{basename}: {basename}.xyz not found in {opt_acc_dir}")

    freq_dir = os.path.join(system_path, "Vacuum", "freq")
    os.makedirs(freq_dir, exist_ok=True)
    
    inp_name_freq = f"{basename}.inp"
    out_name_freq = f"{basename}.out"

    if not is_calculation_done(freq_dir, out_name_freq):
        logging.info(f"{basename}: starting freq (vacuum) in {freq_dir}")
        coords = read_xyz_coordinates(opt_acc_xyz)
        template = os.path.join(INPUT_DIR, "freq.inp")
        inp_path = os.path.join(freq_dir, inp_name_freq)
        generate_input_file(template, coords, inp_path)
        prepare_start_script(freq_dir, inp_name_freq, 'freq')
        job_id = submit_job(freq_dir, basename)
        wait_for_job(job_id)
        check_success(freq_dir, out_name_freq)
        logging.info(f"{basename}: freq (vacuum) completed")
    else:
        logging.info(f"{basename}: freq (vacuum) already done")

    out_file = os.path.join(freq_dir, out_name_freq)
    IR_spectrum_value = extract_IR_spectrum(out_file)
    append_IR_spectrum(basename, IR_spectrum_value)

def H2O_freq(system_path, basename):
    opt_acc_water_dir = os.path.join(system_path, "H2O", "opt_acc")
    opt_acc_water_xyz = os.path.join(opt_acc_water_dir, f"H2O_{basename}.xyz")
    if not os.path.exists(opt_acc_water_xyz):
        raise RuntimeError(f"{basename}: H2O_{basename}.xyz not found in {opt_acc_water_dir}")

    freq_water_dir = os.path.join(system_path, "H2O", "freq")
    os.makedirs(freq_water_dir, exist_ok=True)

    inp_name_freq_water = f"H2O_{basename}.inp"
    out_name_freq_water = f"H2O_{basename}.out"

    if not is_calculation_done(freq_water_dir, out_name_freq_water):
        logging.info(f"{basename}: starting freq (water) in {freq_water_dir}")
        coords = read_xyz_coordinates(opt_acc_water_xyz)
        template = os.path.join(INPUT_DIR, "H2O_freq.inp")
        inp_path = os.path.join(freq_water_dir, inp_name_freq_water)
        generate_input_file(template, coords, inp_path)
        prepare_start_script(freq_water_dir, inp_name_freq_water, 'freq')
        job_id = submit_job(freq_water_dir, 'H2O_' + basename)
        wait_for_job(job_id)
        check_success(freq_water_dir, out_name_freq_water)
        logging.info(f"{basename}: freq (water) completed")
    else:
        logging.info(f"{basename}: freq (water) already done")

    out_file = os.path.join(freq_water_dir, out_name_freq_water)
    IR_spectrum_value = extract_IR_spectrum(out_file)
    append_IR_spectrum(f"H2O_{basename}", IR_spectrum_value)


# --------------------------------------------------------
# Этап 3.2: расчёт частот видимого спектра в вакууме и воде
# --------------------------------------------------------

def td(system_path, basename):
    opt_acc_dir = os.path.join(system_path, "Vacuum", "opt_acc")
    opt_acc_xyz = os.path.join(opt_acc_dir, f"{basename}.xyz") 
    if not os.path.exists(opt_acc_xyz):
        raise RuntimeError(f"{basename}: {basename}.xyz not found in {opt_acc_dir}")

    td_dir = os.path.join(system_path, "Vacuum", "td")
    os.makedirs(td_dir, exist_ok=True)

    inp_name_td = f"{basename}.inp"
    out_name_td = f"{basename}.out"

    if not is_calculation_done(td_dir, out_name_td):
        logging.info(f"{basename}: starting td (vacuum) in {td_dir}")
        coords = read_xyz_coordinates(opt_acc_xyz)
        template = os.path.join(INPUT_DIR, "td.inp")
        inp_path = os.path.join(td_dir, inp_name_td)
        generate_input_file(template, coords, inp_path)
        prepare_start_script(td_dir, inp_name_td, 'td')
        job_id = submit_job(td_dir, basename)
        wait_for_job(job_id)
        check_success(td_dir, out_name_td)
        logging.info(f"{basename}: td (vacuum) completed")
    else:
        logging.info(f"{basename}: td (vacuum) already done")

    out_file = os.path.join(td_dir, out_name_td)
    optical_spectrum_value = extract_optical_spectrum(out_file)
    append_optical_spectrum(basename, optical_spectrum_value)

def H2O_td(system_path, basename):
    opt_acc_water_dir = os.path.join(system_path, "H2O", "opt_acc")
    opt_acc_water_xyz = os.path.join(opt_acc_water_dir, f"H2O_{basename}.xyz")
    if not os.path.exists(opt_acc_water_xyz):
        raise RuntimeError(f"{basename}: H2O_{basename}.xyz not found in {opt_acc_water_dir}")

    td_water_dir = os.path.join(system_path, "H2O", "td")
    os.makedirs(td_water_dir, exist_ok=True)

    inp_name_td_water = f"H2O_{basename}.inp"
    out_name_td_water = f"H2O_{basename}.out"

    if not is_calculation_done(td_water_dir, out_name_td_water): 
        logging.info(f"{basename}: starting td (water) in {td_water_dir}")
        coords = read_xyz_coordinates(opt_acc_water_xyz)
        template = os.path.join(INPUT_DIR, "H2O_td.inp")
        inp_path = os.path.join(td_water_dir, inp_name_td_water)
        generate_input_file(template, coords, inp_path)
        prepare_start_script(td_water_dir, inp_name_td_water, 'td')
        job_id = submit_job(td_water_dir, 'H2O_' + basename)
        wait_for_job(job_id)
        check_success(td_water_dir, out_name_td_water)
        logging.info(f"{basename}: td (water) completed")
    else:
        logging.info(f"{basename}: td (water) already done")

    out_file = os.path.join(td_water_dir, out_name_td_water)
    optical_spectrum_value = extract_optical_spectrum(out_file)
    append_optical_spectrum(f"H2O_{basename}", optical_spectrum_value)


# --------------------------------------------------------
# Вакуум
# --------------------------------------------------------

def Vacuum(system_path, basename):

    # Этап 2: точная оптимизация в вакууме
    
    opt_acc(system_path, basename)
    
    # Этап 3.1: расчёт частот ИК-спектра в вакууме
    # Этап 3.2: расчёт частот видимого спектра в вакууме
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(freq, system_path, basename),
            executor.submit(td, system_path, basename)
        ]
        for f in futures:
            f.result()

# --------------------------------------------------------
# Вода
# --------------------------------------------------------

def H2O(system_path, basename):
    
    # Этап 2: точная оптимизация в воде
    
    H2O_opt_acc(system_path, basename)
    
    # Этап 3.1: расчёт частот ИК-спектра в воде
    # Этап 3.2: расчёт частот видимого спектра в воде
   
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(H2O_freq, system_path, basename),
            executor.submit(H2O_td, system_path, basename)
        ]
        for f in futures:
            f.result()
    
# ------------------------------------------------------------
# Функция, выполняющая все этапы для одной системы (папка system_path)
# ------------------------------------------------------------
def run_system(system_path):
    """
    Полный цикл расчётов для одной системы.
    system_path: путь к папке, содержащей исходный .inp файл.
    """
    # Находим первый .inp файл в папке
    inp_file = [f for f in os.listdir(system_path) if f.endswith('.inp')]
    if not inp_file:
        logging.warning(f"No .inp file found in {system_path}, skipping")
        return
    # Берём первый (можно потребовать ровно один, но пока берём первый)
    inp_filename = inp_file[0]
    basename = os.path.splitext(inp_filename)[0]   # имя без расширения
    base_inp = os.path.join(system_path, inp_filename)

    logging.info(f"Processing system: {system_path} (basename: {basename})")

    # Этап 1: грубая оптимизация
    
    opt(system_path, basename, base_inp)

    # Последующие этапы
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(Vacuum, system_path, basename),
            executor.submit(H2O, system_path, basename)
        ]
        for f in futures:
            f.result()
    

    logging.info(f"{basename}: all calculations finished successfully")


# ------------------------------------------------------------
# Главная функция
# ------------------------------------------------------------
def main():
    # Определяем корневую директорию для поиска систем
    global root_dir
    if len(sys.argv) > 1:
        root_dir = os.path.abspath(sys.argv[1])
    else:
        root_dir = os.getcwd()
        logging.info(f"No root directory provided, using current directory: {root_dir}")

    logging.info(f"Scanning for systems in: {root_dir}")

    # Добавляем файловый обработчик в корневой директории
    global file_handler
    log_path = os.path.join(root_dir, "orca_optimise.log")
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    log.info(f"Log file: {log_path}")
    log.info(f"Scanning for systems in: {root_dir}")
    
    # Сканируем root_dir, ищем подпапки, содержащие .inp файл
    systems = []
    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)
        if os.path.isdir(full_path):
            # Пропускаем служебные папки (input, если вдруг оказалась внутри root_dir)
            if item in ['input']:
                continue 
            pdb_file = [f for f in os.listdir(full_path) if f.endswith('.pdb')]
            pdb_file_path = os.path.join(full_path, pdb_file[0])
            append_fractions(pdb_file_path)
            pdb_to_inp(pdb_file_path)
            # Проверяем наличие .inp файлов внутри
            inp_present = any(f.endswith('.inp') for f in os.listdir(full_path))
            if inp_present:
                systems.append(full_path)

    if not systems:
        logging.error("No directories with .inp files found. Exiting.")
        return

    logging.info(f"Found {len(systems)} systems to process: {[os.path.basename(s) for s in systems]}")

    # Запускаем обработку всех систем параллельно
    with ThreadPoolExecutor(max_workers=len(systems)) as executor:
        futures = [executor.submit(run_system, sys_path) for sys_path in systems]
        for f in futures:
            f.result()   # если хоть одна система упадёт – программа остановится

    logging.info("ALL CALCULATIONS COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()

