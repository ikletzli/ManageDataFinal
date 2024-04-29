This project implements the paper "Unsupervised String Transformation Learning for
Entity Consolidation" found at https://arxiv.org/pdf/1709.10436.pdf.

To run our code, simply run driver.py with any of the following options:


    driver.py [-h] [--address ADDRESS] [--author AUTHOR] [--incgrouping INCGROUPING] [--small] [--verbose]

# Command Line Options:


  **-h, --help**            show this help message and exit

  **--address ADDRESS**     Number of address records to sample (only up to 162 is working due to speed)

  **--author AUTHOR**       Number of author records to sample (only up to 80 is working due to speed)
  
  **--incgrouping INCGROUPING**
                        indicates whether or not incremental grouping should be used, and to what degree

  **--small**               indicates whether or not a small test example should be used to show it working
  
  **--verbose**             indicates whether to print intermediate paths found