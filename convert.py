import csv
from datetime import datetime
import random

def create_new_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        
        # Write header row to the output CSV
        writer.writerow(['sentence', 'language', 'country', 'user_id', 'time'])
        
        # List of username prefixes
        username_prefixes = [
            '@bennyG', '@biggirl', '@Angelina005', '@coolTom', '@sassySam', '@jazzyJeff',
            '@mikeMighty', '@sweetSue', '@craftyCarl', '@vividVic', '@wittyWhit', '@niftyNick',
            '@bubblyBeth', '@jollyJoe', '@quizzicalQuinn', '@zestyZara', '@dreamyDan',
            '@peppyPam', '@rockinRob', '@calmCam', '@hopefulHank', '@livelyLiz', '@merryMira',
            '@neatNate', '@jumpyJill', '@keenKurt', '@braveBri', '@chillChad', '@genialGina',
            '@fancyFran', '@happyHank', '@gracefulGreg'
        ]
        
        for line in infile:
            # Remove the number-dot-space at the beginning of the text
            sentence = line.strip().split('. ', 1)[1]
            
            # Set static values for language and country
            language = 'English'
            country = 'USA'
            
            # Generate random username
            user_id = random.choice(username_prefixes) + str(random.randint(100, 999))
            
            # Generate random timestamp within 2023
            timestamp = random.uniform(datetime(2023, 1, 1).timestamp(), datetime(2023, 12, 31).timestamp())
            
            # Write row to the output CSV
            writer.writerow([sentence, language, country, user_id, timestamp])

# Call the function with the paths to your input and output CSV files
create_new_csv('input.txt', 'output.csv')
