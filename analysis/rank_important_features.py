import csv

filename = 'features_ranked.csv'
# Open the CSV file
rank = {}
with open(filename, 'r') as file:

    # Create a CSV reader object
    csv_reader = csv.reader(file)
    high = len(list(csv_reader)) - 1
    print(high)
    # Read each row of the CSV file
    file.seek(0)
    next(csv_reader)
    i = 0
    for row in csv_reader:
        for elt in row :
            if elt in rank :
                rank[elt] += high-i
            else :
                rank[elt] = high-i
        i+= 1

sorted_dict = dict(sorted(rank.items(), key=lambda item: item[1], reverse=True))

with open('res.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)

    for key, value in sorted_dict.items() :
        # Write elements to the CSV file
        csv_writer.writerow([key, value])