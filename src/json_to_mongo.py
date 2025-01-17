import json
import os
from pymongo import MongoClient
from pathlib import Path
from typing import Union, List, Any
from omegaconf import DictConfig
from tqdm import tqdm
import logging
import boto3
from botocore.handlers import disable_signing
from pymongo.errors import DuplicateKeyError
log = logging.getLogger(__name__)


class MongoDBDataLoader:
    """Class to handle loading JSON data into MongoDB from an NFS storage locker based on batches."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize MongoDB connection.

        Args:
            cfg (DictConfig): OmegaConf DictConfig object containing the configuration settings.
        """
        self.cfg = cfg
        self.client = MongoClient(cfg.mongodb.host,
                                  username=cfg.mongodb.username,
                                  password=cfg.mongodb.password,
                                  authSource=cfg.mongodb.auth_source,
                                  authMechanism=cfg.mongodb.auth_mechanism)
        self.db = self.client[cfg.mongodb.db]
        self.collection = self.db[cfg.mongodb.collection]

        # Root directory of the NFS storage locker
        self.primary_s3_root = Path(cfg.paths.primary_longterm_storage)
        self.secondary_s3_root = Path(cfg.paths.secondary_longterm_storage)

        self.s3_resource = boto3.resource('s3')
        self.s3_resource.meta.client.meta.events.register('choose-signer.s3.*',
                                                          disable_signing)
        self.s3_bucket = self.s3_resource.Bucket(cfg.aws.s3_bucket)
        self.create_id_index("cutouts")


    def s3_folder_exists(self, folder_key):
        if not folder_key.endswith('/'):
            folder_key += '/'
        resp = self.s3_bucket.objects.filter(Prefix=folder_key).limit(1)
        return any(resp)

    def load_json(self, json_path: str) -> Any:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError as e:
            log.exception(f"Error: File not found - {e}")
            raise
        except json.JSONDecodeError as e:
            log.exception(f"Error: Failed to decode JSON - {e}")
            raise
        
    def load_batches_from_yaml(self) -> List[str]:
        """
        Load batch names from a YAML configuration file.

        Args:
            yaml_file_path (Union[str, Path]): Path to the YAML file.
        """
        batches = self.cfg.batches
        log.info(f"Loaded {len(batches)} batches from YAML file.")
        return batches

    def load_json_files_from_batches(self, batches: List[str]) -> None:
        """
        Load JSON files from batch directories in the NFS storage locker and insert them into MongoDB.
        If the batch is not found in the primary storage path, the script checks the alternative storage path.

        Args:
            batches (List[str]): List of batch directories to process.
        """
        # primary_nfs_root_path = Path(self.primary_nfs_root)
        # secondary_nfs_root_path = Path(self.secondary_nfs_root)

        # Loop through all the batch directories
        for batch_name in tqdm(batches):
            batch_dir = os.path.join(self.primary_s3_root, batch_name)
            # Check if the batch exists in the primary storage
            if self.s3_folder_exists(batch_dir):
                json_files = []
                for obj in self.s3_bucket.objects.filter(Prefix=batch_dir):
                    if obj.key.endswith('.json'):
                        json_files.append(obj.key)
                log.info(
                    f"Processing batch '{batch_name}' in primary storage with {len(json_files)} JSON files.")
            else:
                # If not found, check the alternative storage
                batch_dir = os.path.join(self.secondary_s3_root, batch_name)
                if self.s3_folder_exists(batch_dir):
                    json_files = []
                    for obj in self.s3_bucket.objects.filter(Prefix=batch_dir):
                        if obj.key.endswith('.json'):
                            json_files.append(obj.key)
                    log.info(
                        f"Processing batch '{batch_dir}' in alternative storage with {len(json_files)} JSON files.")
                else:
                    log.warning(
                        f"Batch directory '{batch_dir}' not found in either primary or alternative storage.")
                    continue

            for json_file_path in tqdm(json_files):
                self.insert_data_from_file(json_file_path)

    def create_id_index(self, data_type: str) -> None:
        """Create a unique index on the 'cutout_id' field to enforce uniqueness and improve lookup performance."""
        try:
            index_value = "cutout_id" if data_type == 'cutouts' else "image_id"
            self.collection.create_index(index_value, unique=True)
            log.info(f"Unique index on '{index_value}' created successfully.")
        except Exception as e:
            log.exception(f"Failed to create index on '{index_value}': {e}")
            exit()

    def insert_data_from_file(self, json_file_path: Union[str, Path]) -> None:
        """
        Insert data from a JSON file into the MongoDB collection.

        Args:
            json_file_path (Union[str, Path]): Path to the JSON file.
        """
        try:
            # with open(json_file_path) as file:
            #     data = json.load(file)
            data = self.s3_bucket.Object(json_file_path).get()['Body'].read()
            data = json.loads(data.decode('utf-8'))

            # Insert data and handle duplicates
            if isinstance(data, list):
                for item in data:
                    try:
                        self.collection.insert_one(item, bypass_document_validation=False)
                    except DuplicateKeyError as e:
                        log.warning(f"Duplicate entry skipped for {item.get('cutout_id', 'unknown')}: {e}")
            else:
                try:
                    self.collection.insert_one(data, bypass_document_validation=False)
                except DuplicateKeyError as e:
                    log.warning(f"Duplicate entry skipped for {data.get('cutout_id', 'unknown')}: {e}")

        except Exception as e:
            log.exception(f"Failed to insert data from {json_file_path}: {e}")


# Example usage:
def main(cfg: DictConfig) -> None:
    # Set up the loader with MongoDB connection
    data_loader = MongoDBDataLoader(cfg)

    # Load batch names from the YAML file
    batch_names = data_loader.load_batches_from_yaml()

    # Load and insert JSON files from the specified batch directories
    data_loader.load_json_files_from_batches(batch_names)
