# TODO: remove other useless headers
# TODO: dot chart analysis

import csv
import os
from datetime import datetime

useless_headers = ['AIDS', 'Congestive Heart Failure', 'Arterial Occlusion', 'Hemiplegia', 'Leukemia/Lymphoma', 'Liver Disease', 'Solid Tumor', 'Surgery Date', 'Admission Time', 'Discharge Time',
                   'Surgery Start Time', 'Surgery End Time', 'Surgical Time (minutes)', 'Height', 'Weight', 'BMI']

tagging_headers = ['Postoperative Olanzapine',
                   'Postoperative Fluphenazine', 'Postoperative Flupentixol']


def is_positive(row):
    if row['Label'] != '0':
        return True
    return False


def process_bmi(row):
    if row['Height'] == '' or row['Weight'] == '':
        return None
    height = float(row['Height']) / 100
    weight = float(row['Weight'])
    bmi = weight / (height * height)
    return bmi


def process_sur_time(row):
    if row['Surgery Start Time'] == '' or row['Surgery End Time'] == '':
        return None
    surgery_start_time = datetime.strptime(
        row['Surgery Start Time'], '%Y/%m/%d %H:%M')
    surgery_end_time = datetime.strptime(
        row['Surgery End Time'], '%Y/%m/%d %H:%M')
    surgery_time = (surgery_end_time -
                    surgery_start_time).total_seconds() / 60
    if surgery_time < 0:
        surgery_time = -surgery_time
    return surgery_time


def process_others(row):
    others = ['Glomerular Filtration Rate', 'Hemoglobin', 'Neutrophil Absolute Count',
              'Lymphocyte Absolute Count', 'Platelet Count', 'Anesthesia Time', 'Creatinine', 'Albumin']

    processed = False
    if is_positive(row):
        for item in others:
            if row[item] == '':
                row[item] = -1
                processed = True
        return [row, processed]

    for item in others:
        if row[item] == '':
            return [None, False]
        return [row, False]


def process_row(row, headers, positive_processed, negative_dropped):
    [row, processed] = process_others(row)
    if row is None:
        negative_dropped += 1
        return [row, positive_processed, negative_dropped]
    if processed:
        positive_processed += 1

    # Process BMI
    bmi = process_bmi(row)
    if bmi is None and is_positive(row):
        bmi = -1

    # Process surgery time
    surgery_time = process_sur_time(row)
    if surgery_time is None and is_positive(row):
        surgery_time = -1

    if bmi == -1 or surgery_time == -1:
        positive_processed += 1
    if bmi is None or surgery_time is None:
        negative_dropped += 1

    new_row = None
    if bmi is not None and surgery_time is not None:
        new_row = {header: row[header] for header in headers}
        new_row['BMI'] = bmi
        new_row['Surgery Time'] = surgery_time

    return [new_row, positive_processed, negative_dropped]


def display_data(positive_sample_count, negative_sample_count, positive_processed, negative_dropped):
    print('Positive sample count:', positive_sample_count)
    print('Negative sample count:', negative_sample_count)
    print('Positive samples processed:', positive_processed)
    print('Negative samples dropped:', negative_dropped)


with open('datasets/20220328-or-eng-shrink.csv', 'r', encoding='utf-8') as input_file:
    # Read the input file as a dictionary
    reader = csv.DictReader(input_file)
    # Create a list of headers to keep
    headers = [header for header in reader.fieldnames if header not in (
        useless_headers + tagging_headers)]

    # Create a list to hold the output rows
    output_rows = []

    positive_sample_count = 0
    negative_sample_count = 0

    positive_processed = 0
    negative_dropped = 0

    # Loop through each row in the input file
    for row in reader:
        if is_positive(row):
            positive_sample_count += 1
        else:
            negative_sample_count += 1

        # Process the row
        [new_row, positive_processed, negative_dropped] = process_row(
            row, headers, positive_processed, negative_dropped)

        # Discard the row if it doesn't have enough data
        if new_row is None:
            continue

        # Add the new row to the output rows
        output_rows.append(new_row)

output_path = os.path.dirname(__file__) + '\output.csv'

# dump pre-processed data
with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
    # Write the output rows to the output file
    writer = csv.DictWriter(
        output_file, fieldnames=headers + ['Surgery Time'] + ['BMI'])
    writer.writeheader()
    writer.writerows(output_rows)

display_data(positive_sample_count, negative_sample_count,
             positive_processed, negative_dropped)
print('Output written to path:', output_path)
