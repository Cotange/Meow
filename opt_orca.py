import os
import sys
import numpy as np
import pandas as pd
import shutil
import subprocess
import time
import re
import logging
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------------------------
# Настройки
# ------------------------------------------------------------
# Папка с шаблонами находится рядом со скриптом (в той же директории)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")   # шаблоны и start.sh.template
LOG_FILE = os.path.join(SCRIPT_DIR, "opt_orca.log")

# ------------------------------------------------------------
# Логирование
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------

def pdb_to_inp(file_path):

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            data_rows = []
            for line in lines: 
                row_data = [d.strip() for d in line.strip().split()]
                data_rows.append(row_data)

            df = pd.DataFrame(data_rows)

            df[2] = df[2].str[:1]
            columns = [2, 5, 6, 7]
            df = df[columns]
            df = df.rename(columns={2: 'ATOM', 5: 'X', 6: 'Y', 7: 'Z'})

            file_name = os.path.splitext(file_path)[0]
            with open(file_name + '.inp', 'w') as file:
                file.write('! PBE0 D3 def2-SVP Opt' + '\n' + '! TightSCF' + '\n' + '! TightOpt' + '\n' + '* xyz 0 1' + '\n')
                for i in range(len(df['ATOM'])):
                    row_series = df.iloc[i]
                    row_string = ' '.join(row_series.astype(str))
                    file.write(row_string + '\n')
                file.write('*')

    except FileNotFoundError:
        logging.warning(f"File '{file_path}' was not found.")

def ensure_unix_format(filepath):
    """Переводит концы строк в Unix-формат."""
    with open(filepath, 'rb') as f:
        content = f.read().replace(b'\r\n', b'\n')
    with open(filepath, 'wb') as f:
        f.write(content)

def prepare_start_script(dest, inp_name):
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
    out_name = os.path.splitext(inp_name)[0] + '.out'
    new_content = content.replace('orca.out', out_name)
    sh_name = os.path.splitext(inp_name)[0] + '.sh'
    start_script_path = os.path.join(dest, sh_name)
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

def submit_job(folder):
    """Отправляет задание SLURM в папке folder и возвращает job ID."""
    result = subprocess.run(
        ["sbatch", "start.sh"],
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
            break
        time.sleep(20)

def check_success(folder, out_filename):
    """Проверяет наличие нормального завершения в указанном выходном файле."""
    out_file = os.path.join(folder, out_filename)
    if not os.path.exists(out_file):
        raise RuntimeError(f"Output file missing: {out_file}")
    with open(out_file, 'r', errors='ignore') as f:
        content = f.read()
    if "****ORCA TERMINATED NORMALLY****" not in content:
        raise RuntimeError(f"Calculation NOT finished successfully in {folder}")
    logging.info(f"Success: {folder}")

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

    # --------------------------------------------------------
    # Этап 1: грубая оптимизация в Vacuum/opt
    # --------------------------------------------------------
    opt_dir = os.path.join(system_path, "Vacuum", "opt")
    os.makedirs(opt_dir, exist_ok=True)

    inp_name_opt = f"{basename}.inp"
    out_name_opt = f"{basename}.out"

    if not is_calculation_done(opt_dir, out_name_opt):
        logging.info(f"{basename}: starting opt_low in {opt_dir}")
        # Копируем исходный .inp (без изменений) и готовим start.sh
        shutil.copy(base_inp, os.path.join(opt_dir, inp_name_opt))
        prepare_start_script(opt_dir, inp_name_opt)

        # Удаляем исходный .inp файл после копирования
        try:
            os.remove(base_inp)
            logging.info(f"{basename}: removed original input file {base_inp}")
        except OSError as e:
            logging.warning(f"{basename}: could not remove original input file: {e}")
        
        job_id = submit_job(opt_dir)
        wait_for_job(job_id)
        check_success(opt_dir, out_name_opt)
        logging.info(f"{basename}: opt_low completed")
    else:
        logging.info(f"{basename}: opt_low already done, skipping")

    # --------------------------------------------------------
    # Этап 2: точная оптимизация в газе и воде (параллельно)
    # --------------------------------------------------------
    # Координаты берём из opt_low
    opt_xyz = os.path.join(opt_dir, f"{basename}.xyz")
    if not os.path.exists(opt_xyz):
        raise RuntimeError(f"{basename}: orca.xyz not found in {opt_dir}")

    # Подготавливаем папки для точной оптимизации
    opt_acc_dir = os.path.join(system_path, "Vacuum", "opt_acc")
    opt_acc_water_dir = os.path.join(system_path, "H2O", "opt_acc")
    os.makedirs(opt_acc_dir, exist_ok=True)
    os.makedirs(opt_acc_water_dir, exist_ok=True)

    # Имена файлов для газа
    inp_name_acc = f"{basename}.inp"
    out_name_acc = f"{basename}.out"
    # Для воды
    inp_name_acc_water = f"H2O_{basename}.inp"
    out_name_acc_water = f"H2O_{basename}.out"

    def run_opt_acc():
        if is_calculation_done(opt_acc_dir, out_name_acc):
            logging.info(f"{basename}: opt_acc (vacuum) already done")
            return
        logging.info(f"{basename}: starting opt_acc (vacuum) in {opt_acc_dir}")
        coords = read_xyz_coordinates(opt_xyz)
        template = os.path.join(INPUT_DIR, "opt_acc.inp")
        inp_path = os.path.join(opt_acc_dir, inp_name_acc)
        generate_input_file(template, coords, inp_path)
        prepare_start_script(opt_acc_dir, inp_name_acc)
        job_id = submit_job(opt_acc_dir)
        wait_for_job(job_id)
        check_success(opt_acc_dir, out_name_acc)
        logging.info(f"{basename}: opt_acc (vacuum) completed")

    def run_opt_acc_water():
        if is_calculation_done(opt_acc_water_dir, out_name_acc_water):
            logging.info(f"{basename}: opt_acc (water) already done")
            return
        logging.info(f"{basename}: starting opt_acc (water) in {opt_acc_water_dir}")
        coords = read_xyz_coordinates(opt_xyz)
        template = os.path.join(INPUT_DIR, "H2O_opt_acc.inp")
        inp_path = os.path.join(opt_acc_water_dir, inp_name_acc_water)
        generate_input_file(template, coords, inp_path)
        prepare_start_script(opt_acc_water_dir, inp_name_acc_water)
        job_id = submit_job(opt_acc_water_dir)
        wait_for_job(job_id)
        check_success(opt_acc_water_dir, out_name_acc_water)
        logging.info(f"{basename}: opt_acc (water) completed")

    # Параллельный запуск двух оптимизаций
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_opt_acc),
            executor.submit(run_opt_acc_water)
        ]
        for f in futures:
            f.result()   # поднимет исключение при ошибке

    # --------------------------------------------------------
    # Этап 3: расчёт частот в газе и воде (параллельно)
    # --------------------------------------------------------
    # Координаты для газа из opt_acc
    opt_acc_xyz = os.path.join(opt_acc_dir, f"{basename}.xyz") 
    if not os.path.exists(opt_acc_xyz):
        raise RuntimeError(f"{basename}: orca.xyz not found in {opt_acc_dir}")

    # Координаты для воды из opt_acc_water
    opt_acc_water_xyz = os.path.join(opt_acc_water_dir, f"H2O_{basename}.xyz")
    if not os.path.exists(opt_acc_water_xyz):
        raise RuntimeError(f"{basename}: orca.xyz not found in {opt_acc_water_dir}")

    freq_dir = os.path.join(system_path, "Vacuum", "freq")
    freq_water_dir = os.path.join(system_path, "H2O", "freq")
    os.makedirs(freq_dir, exist_ok=True)
    os.makedirs(freq_water_dir, exist_ok=True)

    # Имена файлов для газа
    inp_name_freq = f"{basename}.inp"
    out_name_freq = f"{basename}.out"
    # Для воды
    inp_name_freq_water = f"H2O_{basename}.inp"
    out_name_freq_water = f"H2O_{basename}.out"

    def run_freq():
        if is_calculation_done(freq_dir, out_name_freq):
            logging.info(f"{basename}: freq (vacuum) already done")
            return
        logging.info(f"{basename}: starting freq (vacuum) in {freq_dir}")
        coords = read_xyz_coordinates(opt_acc_xyz)
        template = os.path.join(INPUT_DIR, "freq.inp")
        inp_path = os.path.join(freq_dir, inp_name_freq)
        generate_input_file(template, coords, inp_path)
        prepare_start_script(freq_dir, inp_name_freq)
        job_id = submit_job(freq_dir)
        wait_for_job(job_id)
        check_success(freq_dir, out_name_freq)
        logging.info(f"{basename}: freq (vacuum) completed")

    def run_freq_water():
        if is_calculation_done(freq_water_dir, out_name_freq_water):
            logging.info(f"{basename}: freq (water) already done")
            return
        logging.info(f"{basename}: starting freq (water) in {freq_water_dir}")
        coords = read_xyz_coordinates(opt_acc_water_xyz)
        template = os.path.join(INPUT_DIR, "H2O_freq.inp")
        inp_path = os.path.join(freq_water_dir, inp_name_freq_water)
        generate_input_file(template, coords, inp_path)
        prepare_start_script(freq_water_dir, inp_name_freq_water)
        job_id = submit_job(freq_water_dir)
        wait_for_job(job_id)
        check_success(freq_water_dir, out_name_freq_water)
        logging.info(f"{basename}: freq (water) completed")

    # Параллельный запуск двух расчётов частот
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_freq),
            executor.submit(run_freq_water)
        ]
        for f in futures:
            f.result()

    logging.info(f"{basename}: all calculations finished successfully")

# ------------------------------------------------------------
# Главная функция
# ------------------------------------------------------------
def main():
    # Определяем корневую директорию для поиска систем
    if len(sys.argv) > 1:
        root_dir = os.path.abspath(sys.argv[1])
    else:
        root_dir = os.getcwd()
        logging.info(f"No root directory provided, using current directory: {root_dir}")

    logging.info(f"Scanning for systems in: {root_dir}")

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

