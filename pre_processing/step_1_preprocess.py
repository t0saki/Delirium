import csv
import os
from datetime import datetime

diseases = ['Dementia', 'Chronic Obstructive Pulmonary Disease', 'Hypertension', 'Coronary Artery Disease', 'Alzheimer\'s Disease',
            'Creatinine Greater Than 176.8', 'Albumin Less Than 34', 'Chest X-ray Indicating Atelectasis/Pulmonary Infiltration']

# the label is removed from the headers, and will be added back later
useless_headers = ['AIDS', 'Congestive Heart Failure', 'Arterial Occlusion', 'Hemiplegia', 'Leukemia/Lymphoma', 'Liver Disease', 'Hypertension', 'Coronary Artery Disease', 'Prolonged Length of Stay (Postoperative Hospital Stay > 7 days)',
                   'Solid Tumor', 'Creatinine', 'Peptic Ulcer', 'Imaging - Exudation 0 No 1 Yes', 'C-Reactive Protein (CRP)', 'Hematocrit', 'Glomerular Filtration Rate', 'Chest X-ray Indicating Atelectasis/Pulmonary Infiltration',
                   'In-Hospital Mortality', 'Use of Whole Blood', 'Whole Blood', 'Surgery Date', 'Admission Time', 'Discharge Time', 'Chronic Obstructive Pulmonary Disease', 'Plasma', 'Red Blood Cells',
                   'Surgery Start Time', 'Intraoperative Lactate', 'Intraoperative Blood Loss', 'Surgery End Time', 'Surgical Time (minutes)', 'Height', 'Weight', 'BMI', 'Label']

# 只要有一个疾病，就认为是生病了
tagging_headers_disease = ['Postoperative Olanzapine',
                           'Postoperative Fluphenazine', 'Postoperative Flupentixol']


def is_positive(row):
    if row['Label'] != '0':
        return True
    return False


tagging_headers_blood = ['Use of Plasma',
                         'Use of Platelets', 'Use of Red Blood Cells']


def is_positive_blood(row):
    for header in tagging_headers_blood:
        if row[header] != '0':
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
    if surgery_time == 0:
        return None
    return surgery_time


def process_others(row):
    others = ['Glomerular Filtration Rate', 'Hemoglobin', 'Neutrophil Absolute Count',
              'Lymphocyte Absolute Count', 'Platelet Count', 'Anesthesia Time', 'Creatinine', 'Albumin']

    processed = False

    for item in others:
        if row[item] == '':
            row[item] = -1
            processed = True
    return [row, processed]


# 最主要的方法，处理每一行数据
def process_row(row, headers, positive_processed, negative_dropped):
    [row, processed] = process_others(row)

    # if row is None:
    #     negative_dropped += 1
    #     return [row, positive_processed, negative_dropped]

    # 特殊处理BMI
    bmi = process_bmi(row)
    if bmi is None:
        # if bmi is None and is_positive(row):
        bmi = -1

    # 特殊处理手术时间
    surgery_time = process_sur_time(row)
    if surgery_time is None:
        # if surgery_time is None and is_positive(row):
        surgery_time = -1

    if bmi == -1 or surgery_time == -1 or processed:
        positive_processed += 1
    # if bmi is None or surgery_time is None:
    #     negative_dropped += 1

    # disease_order = process_diseases(row)

    new_row = None
    if bmi is not None and surgery_time is not None:
        new_row = {header: row[header] for header in headers}
        new_row['Surgery Time'] = surgery_time
        new_row['BMI'] = bmi
        # new_row['Disease Order'] = disease_order
        new_row['Use of Blood'] = is_positive_blood(row)
        new_row['Label'] = row['Label']

    return [new_row, positive_processed, negative_dropped]


def display_data(positive_sample_count, negative_sample_count, positive_processed, negative_dropped):
    print('Positive sample count:', positive_sample_count)
    print('Negative sample count:', negative_sample_count)
    print('Positive samples processed:', positive_processed)
    print('Negative samples dropped:', negative_dropped)


# 程序入口

# 读取数据
with open('datasets/20220328-or-eng-shrink.csv', 'r', encoding='utf-8') as input_file:
    # 根据路径读取数据·
    reader = csv.DictReader(input_file)

    # 检测是否有Header，若没有表头则退出 -> array
    field_names = reader.fieldnames
    if field_names is None:
        print('No data in input file')
        exit(1)

    # 把需要导出的Header提取出来，除了要舍弃的和Tag(是否生病)
    headers = [header for header in field_names if header not in (
        useless_headers + tagging_headers_disease + tagging_headers_blood)]

    # 用来存放处理后的每行数据
    output_rows = []

    # 统计信息
    positive_sample_count = 0
    negative_sample_count = 0
    positive_processed = 0
    negative_dropped = 0

    # 遍历每一行数据
    for row in reader:
        # 检测是否生病
        if is_positive(row):
            positive_sample_count += 1
        else:
            negative_sample_count += 1

        # 读取每一行数据，返回处理后的数据，以及统计信息·
        [new_row, positive_processed, negative_dropped] = process_row(
            row, headers, positive_processed, negative_dropped)

        # # Discard the row if it doesn't have enough data
        # if new_row is None:
        #     continue

        # Add the new row to the output rows
        output_rows.append(new_row)

output_path = os.path.dirname(__file__) + '\\output.csv'

# dump pre-processed data
with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
    # Write the output rows to the output file
    writer = csv.DictWriter(
        output_file, fieldnames=headers + ['Surgery Time'] + ['BMI'] + ['Use of Blood'] + ['Label'])
    writer.writeheader()
    writer.writerows(output_rows)

display_data(positive_sample_count, negative_sample_count,
             positive_processed, negative_dropped)
print('Output written to path:', output_path)
