import csv
from datetime import datetime

# Open the input file
with open('datasets/20220328-or-eng-shrink.csv', 'r', encoding='utf-8') as input_file:
    # Read the input file as a dictionary
    reader = csv.DictReader(input_file)
    # Create a list of headers to keep
    headers = [header for header in reader.fieldnames if header not in [
        'AIDS', 'Surgery Date', 'Admission Time', 'Discharge Time', 'Surgery Start Time', 'Surgery End Time', 'Surgical Time (minutes)', 'Height', 'Weight', 'BMI']]
    # Create a list to hold the output rows
    output_rows = []
    # Initialize the average surgery time and time elapsed
    avg_surgery_time = 0
    avg_bmi = 0
    valid_bmi_count = 0
    valid_surgery_time_count = 0
    # Loop through each row in the input file
    for row in reader:
        if row['Height'] == '' or row['Weight'] == '':
            bmi = avg_bmi
        else:
            valid_bmi_count += 1
            # Calculate the BMI
            height = float(row['Height']) / 100
            weight = float(row['Weight'])
            bmi = weight / (height * height)
            avg_bmi = (avg_bmi * (valid_bmi_count - 1) + bmi) / valid_bmi_count

        if row['Surgery Start Time'] == '' or row['Surgery End Time'] == '':
            surgery_time = avg_surgery_time
        else:
            valid_surgery_time_count += 1
            # Calculate the surgery time in minutes
            surgery_start_time = datetime.strptime(
                row['Surgery Start Time'], '%Y/%m/%d %H:%M')
            surgery_end_time = datetime.strptime(
                row['Surgery End Time'], '%Y/%m/%d %H:%M')
            surgery_time = (surgery_end_time -
                            surgery_start_time).total_seconds() / 60
            if surgery_time < 0:
                surgery_time = -surgery_time
                avg_surgery_time = (
                    avg_surgery_time * (valid_surgery_time_count - 1) + surgery_time) / valid_surgery_time_count

        # Create a new row with the calculated values
        new_row = {header: row[header] for header in headers}
        new_row['Surgery Time'] = surgery_time
        new_row['BMI'] = bmi
        # Add the new row to the output rows
        output_rows.append(new_row)

# Open the output file
with open('output.csv', 'w', newline='', encoding='utf-8') as output_file:
    # Write the output rows to the output file
    writer = csv.DictWriter(
        output_file, fieldnames=headers + ['Surgery Time'] + ['BMI'])
    writer.writeheader()
    writer.writerows(output_rows)

# Print a success message
print('Output file written successfully.')
