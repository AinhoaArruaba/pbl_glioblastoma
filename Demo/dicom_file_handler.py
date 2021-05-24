import os
import numpy as np
import pydicom as dicom
from datetime import datetime
import re
from pprint import pprint


def load_subject_scan(scan_path):
    slices = [dicom.read_file(scan_path + '/' + s)
              for s in os.listdir(scan_path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(
            slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def save_subject_scan(raw_slices, subject, scan_path, scan_type, new_folder):
    slices = [dicom.read_file(scan_path + '/' + s)
              for s in os.listdir(scan_path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(
            slices[0].SliceLocation - slices[1].SliceLocation)

    if not os.path.exists(os.path.join(new_folder, subject, scan_type)):
        os.mkdir(os.path.join(new_folder, subject, scan_type))

    for index in enumerate(slices):
        slices[index[0]].SliceThickness = slice_thickness
        slices[index[0]].PixelData = raw_slices[index[0], :, :].tobytes()

        slices[index[0]].save_as(os.path.join(
            new_folder, subject, scan_type, 'slice_' + str(index[0]) + '.dcm'))

def save_subject_scan_seg(raw_slices, subject, scan_path, scan_type, new_folder):
    slices = [dicom.read_file(scan_path + '/' + s)
              for s in os.listdir(scan_path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(
            slices[0].SliceLocation - slices[1].SliceLocation)

    if not os.path.exists(os.path.join(new_folder, subject, scan_type)):
        os.mkdir(os.path.join(new_folder, subject, scan_type))

    for index in enumerate(slices):
        slices[index[0]].SliceThickness = slice_thickness
        slices[index[0]].PixelData = raw_slices[index[0]].tobytes()

        slices[index[0]].save_as(os.path.join(
            new_folder, subject, scan_type, 'slice_' + str(index[0]) + '.dcm'))


def extract_raw_img_array(slices):
    ref_slice = slices[0]
    pixel_dims = (len(slices), int(ref_slice.Rows), int(ref_slice.Columns))
    # pixel_spaces = (float(ref_slice.PixelSpacing[0]), float(
    #     ref_slice.PixelSpacing[1]), float(ref_slice.SliceThickness))

    raw_images = np.zeros(pixel_dims, dtype=ref_slice.pixel_array.dtype)

    # loop through all the DICOM files
    for index, dcm_file in enumerate(slices):
        # store the raw image data
        raw_images[index, :, :] = dcm_file.pixel_array

    return raw_images


def identify_scan_folder(path):
    scans = []
    subject_studies = os.listdir(path) if os.path.isdir(path) else []
    study_folder = None

    if subject_studies:
        if len(subject_studies) > 0:
            dates = []
            for index, study in enumerate(subject_studies):
                match = re.search(r'\d{2}-\d{2}-\d{4}', study)
                if match:
                    dates.append([index, datetime.strptime(
                        match.group(), '%m-%d-%Y').date()])
            dates.sort(key=lambda date: date[1], reverse=False)
            study_folder = os.path.join(path, subject_studies[dates[0][0]])

        study_scans = os.listdir(study_folder)
        for index, scan in enumerate(study_scans):
            if not '.ds_store' in scan.lower():
                scans.append(scan)
    return study_folder, scans


def get_subject_list(path):
    folders = os.listdir(path)
    subjects = []
    for f in folders:
        if os.path.isdir(os.path.join(path, f)):
            subjects.append(f)
    return subjects


def write_dicom(new_img, dicom_data, folder, s):
    # Create new folder for new slices
    dir_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'database_noskull')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    dir_path = os.path.join(dir_path, s)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    dir_path = os.path.join(dir_path, folder)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    new_dicom_data = dicom_data
    new_img = np.array(new_img)
    for index, slc in enumerate(new_dicom_data):
        slc.PixelData = new_img[index, :, :]
        slc.save_as(os.path.join(dir_path, os.path.basename(slc.filename)))


if __name__ == "__main__":
    base_path = "/Users/ainhoaarruabarrenaortiz/Documents/Master/PBL/Datasets/Data/CPTAC-GBM"
    subjects = [os.path.join(base_path, subject)
                for subject in sorted(os.listdir(base_path))]

    data = {os.path.basename(subject): load_subject_scan(subject)
            for subject in subjects}

    pprint(data, width=100)

    print(len(data))
