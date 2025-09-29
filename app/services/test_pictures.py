from app.services import data_loader
from app.services import thread_service
import yaml
config = yaml.safe_load(open('configs/config.yaml', 'r'))
MESSAGES_FILE_PATH ='C:\\VSCode\\scraping\\data\\discord_messages_names_gcs_pics.xlsx'
messages_df, _ = data_loader.load_and_preprocess_data(MESSAGES_FILE_PATH)
df = messages_df[messages_df['Author ID'] == messages_df[messages_df['Attachments'] != "No attachments"]['Author ID'].iloc[5]]
result_path = thread_service.first_thread_gathering(df,'test_first_group', config)
print(result_path)