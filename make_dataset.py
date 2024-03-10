import os
import pandas as pd
from tqdm import tqdm
import coloredlogs, logging
from decouple import config
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', 'INFO'), logger=logger)

maildir_path = './data/raw/maildir'
rows = []


def extract_email_contents(email_file_path, rows, label_folder, person_folder):
    if email_file_path != '.DS_Store':
        # Extract the label from the folder name
        label = label_folder
        
        try:
            with open(email_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                email_contents = file.read()
        except Exception as e:
            logger.error(f'Error reading email file: {email_file_path}, Error: {e}')
            import ipdb; ipdb.set_trace()
            email_contents = None

        # Create the row
        row = {
            'person': person_folder,
            'label': label,
            'email_file': email_file_path,
            'email_contents': email_contents
        }
        rows.append(row)

def traverse_directory(directory_path, rows, label_folder, person_folder):
    if os.path.isdir(directory_path):
        for file_or_dir in os.listdir(directory_path):
            file_or_dir_path = os.path.join(directory_path, file_or_dir)
            if os.path.isdir(file_or_dir_path):
                traverse_directory(file_or_dir_path, rows, label_folder, person_folder)
            else:
                extract_email_contents(file_or_dir_path, rows, label_folder, person_folder)
    else:
        extract_email_contents(directory_path, rows, label_folder, person_folder)

# Iterate over each person's folder
logger.info('Iterating over each person\'s folder...')
for person_folder in tqdm(os.listdir(maildir_path)):
    person_folder_path = os.path.join(maildir_path, person_folder)
    
    # Iterate over each label folder within the person's folder
    if os.path.isdir(person_folder_path):
        for label_folder in os.listdir(person_folder_path):
            label_folder_path = os.path.join(person_folder_path, label_folder)
            traverse_directory(label_folder_path, rows, label_folder, person_folder)
                
            
logger.info('Done iterating over each person\'s folder. ✅')

logger.info('Creating a dataframe and saving to CSV...')
# Create a dataframe from the labels list
df = pd.DataFrame(rows)

logger.info(f'Dataframe: \n{df.head()}')

# Save the dataframe to a CSV file
df.to_csv('./data/processed/enron_emails.csv', index=False)

logger.info('Done creating a dataframe and saving to CSV. ✅')