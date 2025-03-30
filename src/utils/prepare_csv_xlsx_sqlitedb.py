import os 
import pandas as pd 
import yaml 
from pyprojroot import here   # We use this to find the root directory ot our project 
# here() returns a function that constructs absolute paths relative to the project root directory 
from dotenv import load_dotenv
import chromadb
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(dotenv_path=here(".env"))




class LoadConfig:
    def __init__(self) -> None:
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        self.load_directories(app_config=app_config)
        self.load_llm_configs(app_config=app_config)
        self.load_openai_models()
        self.load_chroma_client()
        self.load_rag_config(app_config=app_config)

        # If we  want to clean up the upload csv SQL DB on every fresh run of the chatbot we can uncomment the next line . (if it exists)
        # self.remove_directory(self.uploaded_files_sqldb_directory)



    def load_directories(self, app_config):
        self.stored_csv_xlsx_directory = here(
            app_config["directories"]["stored_csv_xlsx_directory"])
        self.sqldb_directory = str(here(
            app_config["directories"]["sqldb_directory"]))
        self.uploaded_files_sqldb_directory = str(here(
            app_config["directories"]["uploaded_files_sqldb_directory"]))
        self.stored_csv_xlsx_sqldb_directory = str(here(
            app_config["directories"]["stored_csv_xlsx_sqldb_directory"]))
        self.persist_directory = app_config["directories"]["persist_directory"]




    def load_llm_configs(self, app_config):
        self.model_name = os.getenv("gpt_deployment_name")
        self.agent_llm_system_role = app_config["llm_config"]["agent_llm_system_role"]
        self.rag_llm_system_role = app_config["llm_config"]["rag_llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]
        self.embedding_model_name = os.getenv("embed_deployment_name")




    def load_openai_models(self):
        openai_api_key = os.environ["OPENAI_API_KEY"]

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)

        #  callable for generating completions
        self.chat = lambda messages: self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )

        #  callable for embeddings
        self.embed = lambda input_text: self.openai_client.embeddings.create(
            model=self.embedding_model_name,
            input=input_text
        ).data[0].embedding



    def load_chroma_client(self):
        self.chroma_client = chromadb.PersistentClient(
            path=str(here(self.persist_directory)))

    def load_rag_config(self, app_config):
        self.collection_name = app_config["rag_config"]["collection_name"]
        self.top_k = app_config["rag_config"]["top_k"]

    def remove_directory(self, directory_path: str):
        """
        Removes the specified directory.

        Parameters:
            directory_path (str): The path of the directory to be removed.

        Raises:
            OSError: If an error occurs during the directory removal process.

        Returns:
            None
        """
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(
                    f"The directory '{directory_path}' has been successfully removed.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"The directory '{directory_path}' does not exist.")
