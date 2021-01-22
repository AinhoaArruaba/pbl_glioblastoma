import time
import os


def wait_file_created(path, time_wait=10):
    time_counter = 0
    while not os.path.exists(path):
        time.sleep(1)
        time_counter += 1
        if time_counter > time_wait:
            break

    return not time_counter > time_wait


def remove_files(file_list):
    for f in file_list:
        if os.path.exists(f):
            os.remove(f)
