# Housekeeping for metadata

from os.path import join

# Path to the root data directory
ROOT_DATA_DIR = "/mnt/DataDrive1/shaharia/Test_Data"

# clean up metadata
CLEANUP_METADATA = join(ROOT_DATA_DIR, "data_cleanup.xlsx")

# cleaned up metadata
try:
    CLEANED_METADATA = join(ROOT_DATA_DIR, "data_cleanup_parsed.xlsx")
except FileNotFoundError:
    CLEANED_METADATA = None
    print("CLEANED_METADATA not found")
