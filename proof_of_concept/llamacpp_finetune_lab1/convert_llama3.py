import csv

def convert_csv_to_output(input_file, output_file):
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the first row

        with open(output_file, 'w') as outfile:
            for row in reader:
                column_1 = row[0]
                column_2 = row[1]
                outfile.write(f"<SFT><|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a high school science teacher. Explain concepts in very simple English.<|eot_id|><|start_header_id|>user<|end_header_id|>{column_1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{column_2}")

# Testing the function
input_file = 'train.csv'  # Replace with your input file name
output_file = 'train.txt'  # Replace with your desired output file name

convert_csv_to_output(input_file, output_file)
