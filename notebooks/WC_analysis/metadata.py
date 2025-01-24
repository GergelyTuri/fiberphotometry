# Housekeeping for metadata

from os.path import join, exists 

# Path to the root data directory
# ROOT_DATA_DIR = "/mnt/DataDrive1/shaharia/Test_Data"


# clean up metadata
#CLEANUP_METADATA = join(ROOT_DATA_DIR, "data_cleanup.xlsx")

# cleaned up metadata
#try:
    #CLEANED_METADATA = join(ROOT_DATA_DIR, "data_cleanup_parsed.xlsx")
#except FileNotFoundError:
    #CLEANED_METADATA = None
    #print("CLEANED_METADATA not found")

# good quality metadata
#try:
    #GOOD_METADATA = join(ROOT_DATA_DIR, "data_cleanup_parsed_good.xlsx")
#except FileNotFoundError:
    #GOOD_METADATA = None
    #print("GOOD_METADATA not found")


##############################################################################

# Modified script

# Path to the root data directory
ROOT_DATA_DIR = "C:/Users/khansha/Test_Data"

# clean up metadata
CLEANUP_METADATA = join(ROOT_DATA_DIR, "data_cleanup.xlsx")

# cleaned up metadata
CLEANED_METADATA = join(ROOT_DATA_DIR, "data_cleanup_parsed.xlsx")
if not exists(CLEANED_METADATA):
    CLEANED_METADATA = None
    print("CLEANED_METADATA not found")

# good quality metadata
GOOD_METADATA = join(ROOT_DATA_DIR, "data_cleanup_parsed_good.xlsx")
if not exists(GOOD_METADATA):
    GOOD_METADATA = None
    print("GOOD_METADATA not found")
