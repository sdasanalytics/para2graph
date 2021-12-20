#----------------------------#
# Author: Surjit Das
# Email: surjitdas@gmail.com
# Program: artmind
#----------------------------#

from loguru import logger as log
import sys
import constants as C
from textprocessor import TextProcessor
from tqdm import tqdm
from functools import partialmethod

log.remove() #removes default handlers
log.level("D_DEBUG", no=33, icon="🤖", color="<blue>")
log.add(C.LOG_PATH, backtrace=True, diagnose=True, level="DEBUG")
log.__class__.d_debug = partialmethod(log.__class__.log, "D_DEBUG")

@log.catch
def main():
    if len(sys.argv) < 2:
        print("Please provide params:\n1) interaction_type (inline|file)\n2) if param1 is file - provide full filepath")
        exit(0)
    tp = TextProcessor("truncate") # ToDo: add this as a cmd line parameter

    if sys.argv[1] == "file":
        if len(sys.argv) != 3:
            print("Please provide full filename as 2nd parameter")
            exit(0)
        
        with open(sys.argv[2]) as fp: 
            lines = fp.readlines() 
            for line in tqdm(lines, desc="Processing sentences"):
                log.info(f"Processing line: {line}")
                tp.execute(line.strip())
            # tp.save_graph(mode="overwrite") <-- Change
        log.info("Done")
    else:
        text=input("Para: ")

        while(text!="/stop"):
            tp.execute(text) 
            print("Done...")
            text = input("Para: ")
        # tp.save_graph(mode="overwrite") <-- Change

if __name__=="__main__":
    main()
