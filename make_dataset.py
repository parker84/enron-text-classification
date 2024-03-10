import os
import pandas as pd
from tqdm import tqdm
import coloredlogs, logging
from decouple import config
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', 'INFO'), logger=logger)


def extract_email_contents(email_file_path, rows, label_folder, person_folder):
    if email_file_path != '.DS_Store':
        label = label_folder
        try:
            with open(email_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                email_contents = file.read()
        except Exception as e:
            logger.error(f'Error reading email file: {email_file_path}, Error: {e}')
            email_contents = None
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

def build_tabular_dataset(maildir_path):
    rows = []
    logger.info('Iterating over each person\'s folder...')
    for person_folder in tqdm(os.listdir(maildir_path)):
        person_folder_path = os.path.join(maildir_path, person_folder)
        if os.path.isdir(person_folder_path):
            for label_folder in os.listdir(person_folder_path):
                label_folder_path = os.path.join(person_folder_path, label_folder)
                traverse_directory(label_folder_path, rows, label_folder, person_folder)
    logger.info('Done iterating over each person\'s folder. ✅')
    logger.info('Creating a dataframe and saving to CSV...')
    df = pd.DataFrame(rows)
    logger.info(f'Dataframe Shape: {df.shape}')
    logger.info(f'Dataframe Head: \n{df.head()}')
    logger.info(f'Dataframe Tail: \n{df.tail()}')
    logger.info('Done creating a dataframe. ✅')
    return df

def build_ads(df):
    logger.info('Building Ads...')
    labels = [
        'personal', 'meetings', 'logistics', 'resumes', 'gas', 'europe', 'projects', 'bankrupt', 'private_folders',
        'california', 'power', 'hr', 'personalfolder', 'enron_news', 'universities',
        'india', 'europe', 'enron_power', 'california_issues', 'canada', 'rainy_day', 'blue_dog', 'personnel'
    ]
    unique_emails = df[['email_contents']].drop_duplicates()
    unique_emails['email_id'] = range(1, len(unique_emails) + 1)
    df = df.merge(unique_emails, on='email_contents', how='left')
    df['label_value'] = 1
    ads = df[df.label.isin(labels)].pivot(index=['email_id', 'email_contents'], columns='label', values='label_value').reset_index()
    ads.fillna(0, inplace=True)
    logger.info(f'Ads Shape: {ads.shape}')
    logger.info(f'Ads Head: \n{ads.head()}')
    logger.info(f'Ads Tail: \n{ads.tail()}')
    logger.info(f'Ads Description: \n{ads.describe()}')
    logger.info(f'Ads Columns: \n{ads.columns}')
    logger.info('Done building Ads. ✅')
    return ads

def train_validate_test_split(df):
    logger.info('Splitting the dataset into Train, Validate and Test...')
    train = df.sample(frac=0.8, random_state=200)
    validate = df.drop(train.index).sample(frac=0.5, random_state=200)
    test = df.drop(train.index).drop(validate.index)
    logger.info(f'Train Shape: {train.shape}')
    logger.info(f'Validate Shape: {validate.shape}')
    logger.info(f'Test Shape: {test.shape}')
    logger.info('Done Splitting the dataset into Train, Validate and Test. ✅')
    return train, validate, test

if __name__ == '__main__':
    maildir_path = './data/raw/maildir'
    df = build_tabular_dataset(maildir_path)
    logger.info('Saving Enron Emails to CSV...')
    df.to_csv('./data/processed/enron_emails.csv', index=True)
    logger.info('Done Saving Enron Emails to CSV. ✅')
    ads = build_ads(df)
    logger.info('Saving Ads to CSV...')
    ads.to_csv('./data/processed/ads.csv', index=True)
    logger.info('Done Saving Ads to CSV. ✅')
    train, validate, test = train_validate_test_split(ads)
    logger.info('Saving Train, Validate and Test to CSV...')
    train.to_csv('./data/processed/train.csv', index=True)
    validate.to_csv('./data/processed/validate.csv', index=True)
    test.to_csv('./data/processed/test.csv', index=True)
    logger.info('Done Saving Train, Validate and Test to CSV. ✅')